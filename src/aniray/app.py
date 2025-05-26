from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.responses import HTMLResponse
from contextlib import asynccontextmanager
import xarray as xr
import numpy as np
import json
from typing import List, Dict, Any, Optional, Tuple
import asyncio
from pathlib import Path
import ray
from functools import lru_cache
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables to store the dataset
dataset = None
dataset_info = None

@ray.remote
class DatasetWorker:
    """Ray actor for handling dataset operations"""
    
    def __init__(self, zarr_path: str):
        self.dataset = None
        self.zarr_path = zarr_path
        self.load_dataset()
    
    def load_dataset(self):
        """Load the zarr dataset"""
        try:
            self.dataset = xr.open_zarr(self.zarr_path, chunks='auto')
            logger.info(f"Ray worker loaded dataset from {self.zarr_path}")
            return True
        except Exception as e:
            logger.error(f"Ray worker failed to load dataset: {e}")
            return False
    
    def _get_coord_info(self, coord):
        """Extract coordinate information with graceful handling of non-numeric types"""
        coord_info = {
            'values': coord.values.tolist() if coord.size < 1000 else coord.values[::max(1, coord.size//100)].tolist(),
            'size': int(coord.size),
            'dtype': str(coord.dtype),
            'is_numeric': False
        }
        
        # Try to determine if coordinate is numeric and get min/max
        try:
            # Check if the coordinate is numeric
            if np.issubdtype(coord.dtype, np.number):
                coord_info['is_numeric'] = True
                coord_info['min'] = float(coord.min().values)
                coord_info['max'] = float(coord.max().values)
            else:
                # Non-numeric coordinate (strings, datetime, etc.)
                coord_info['is_numeric'] = False
                coord_info['min'] = None
                coord_info['max'] = None
                # For string coordinates, provide some sample values
                if coord.size > 0:
                    sample_size = min(5, coord.size)
                    coord_info['sample_values'] = coord.values[:sample_size].tolist()
                    
        except (ValueError, TypeError) as e:
            # Fallback for any problematic coordinates
            logger.warning(f"Could not process coordinate {coord.name}: {e}")
            coord_info['is_numeric'] = False
            coord_info['min'] = None
            coord_info['max'] = None
            coord_info['sample_values'] = ["<unable to read>"]
            
        return coord_info
    
    def get_dataset_info(self):
        """Extract dataset metadata"""
        if self.dataset is None:
            return None
            
        return {
            'dimensions': list(self.dataset.dims.keys()),
            'dimension_sizes': {dim: int(size) for dim, size in self.dataset.dims.items()},
            'data_vars': list(self.dataset.data_vars.keys()),
            'coords': {name: self._get_coord_info(coord) for name, coord in self.dataset.coords.items()},
            'shape': [int(self.dataset.dims[dim]) for dim in self.dataset.dims.keys()],
            'data_var_info': {var: {
                'dtype': str(self.dataset[var].dtype),
                'shape': list(self.dataset[var].shape)
            } for var in self.dataset.data_vars.keys()}
        }
    
    def get_data_slice(self, data_var: str, selection: Dict, downsample_factor: int = 1):
        """Get a data slice with optional downsampling"""
        if self.dataset is None:
            raise ValueError("Dataset not loaded")
        
        try:
            # Get the data slice
            data_slice = self.dataset[data_var].isel(selection)
            
            # Apply downsampling if requested
            if downsample_factor > 1:
                # Downsample each displayed dimension
                new_selection = {}
                for dim, sel in selection.items():
                    if isinstance(sel, slice) and sel == slice(None):
                        # This is a displayed dimension, downsample it
                        dim_size = self.dataset.dims[dim]
                        new_selection[dim] = slice(0, dim_size, downsample_factor)
                    else:
                        new_selection[dim] = sel
                
                data_slice = self.dataset[data_var].isel(new_selection)
            
            # Convert to numpy and handle NaN values
            data_array = data_slice.values
            data_array = np.nan_to_num(data_array, nan=0.0)
            
            return data_array
            
        except Exception as e:
            logger.error(f"Error getting data slice: {e}")
            raise
    
    def get_coordinates(self, axis_names: List[str], downsample_factor: int = 1):
        """Get coordinate values for specified axes"""
        coords = {}
        for axis in axis_names:
            if axis in self.dataset.coords:
                coord_values = self.dataset.coords[axis].values
                if downsample_factor > 1:
                    coord_values = coord_values[::downsample_factor]
                coords[axis] = coord_values.tolist()
        return coords

@ray.remote
def process_animation_frame(worker_ref, data_var: str, selection: Dict, 
                          animate_axis: str, frame_index: int, 
                          downsample_factor: int = 1) -> Dict:
    """Process a single animation frame"""
    try:
        # Update selection for this frame
        frame_selection = selection.copy()
        frame_selection[animate_axis] = frame_index
        
        # Get data slice
        data_array = ray.get(worker_ref.get_data_slice.remote(
            data_var, frame_selection, downsample_factor
        ))
        
        # Get coordinate value for this frame
        coords = ray.get(worker_ref.get_coordinates.remote([animate_axis]))
        animate_value = coords[animate_axis][frame_index] if downsample_factor == 1 else coords[animate_axis][frame_index // downsample_factor]
        
        return {
            "type": "frame",
            "frame_index": frame_index,
            "data": data_array.tolist(),
            "animate_axis_value": float(animate_value),
            "downsample_factor": downsample_factor
        }
        
    except Exception as e:
        logger.error(f"Error processing frame {frame_index}: {e}")
        return {"error": f"Frame processing error: {str(e)}"}

def calculate_optimal_downsample(shape: Tuple[int, ...], target_size: int = 500) -> int:
    """Calculate optimal downsampling factor to keep data under target size"""
    max_dim = max(shape)
    if max_dim <= target_size:
        return 1
    return max(1, max_dim // target_size)

@lru_cache(maxsize=32)
def get_downsample_options(dim_size: int) -> List[Dict]:
    """Get available downsampling options for a dimension"""
    options = [{"factor": 1, "label": "Full Resolution", "size": dim_size}]
    
    for factor in [2, 4, 8, 16]:
        if dim_size // factor >= 10:  # Don't go below 10 points
            new_size = dim_size // factor
            options.append({
                "factor": factor, 
                "label": f"1/{factor} Resolution", 
                "size": new_size
            })
    
    return options

def load_dataset(zarr_path: str):
    """Initialize Ray and load the dataset"""
    global dataset_info
    
    try:
        # Initialize Ray if not already done
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
            logger.info("Ray initialized successfully")
        
        # Create dataset worker
        worker = DatasetWorker.remote(zarr_path)
        
        # Get dataset info
        dataset_info = ray.get(worker.get_dataset_info.remote())
        
        if dataset_info is None:
            logger.error("Failed to load dataset")
            return None
        
        logger.info(f"Dataset loaded successfully!")
        logger.info(f"Dimensions: {dataset_info['dimensions']}")
        logger.info(f"Shape: {dataset_info['shape']}")
        logger.info(f"Data variables: {dataset_info['data_vars']}")
        
        return worker
        
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        dataset_info = None
        return None

# Global Ray worker reference
dataset_worker = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global dataset_worker
    # Startup - modify this path to point to your zarr dataset
    zarr_path = "/scratch/bester/breifast_test/obs-omcen/cubes/cube-1.zarr"  # MODIFY THIS PATH
    logger.info(f"Attempting to load dataset from: {zarr_path}")
    dataset_worker = load_dataset(zarr_path)
    yield
    # Shutdown
    if ray.is_initialized():
        ray.shutdown()
        logger.info("Ray shutdown complete")

# Create FastAPI app with lifespan
app = FastAPI(lifespan=lifespan)

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        try:
            await websocket.send_text(message)
        except Exception as e:
            logger.error(f"Error sending message: {e}")

manager = ConnectionManager()

@app.get("/")
async def get():
    return HTMLResponse(content=open("index.html").read(), media_type="text/html")

@app.get("/dataset-info")
async def get_dataset_info():
    """Return metadata about the loaded dataset"""
    if dataset_info is None:
        return {"error": "No dataset loaded"}
    
    # Add downsampling options for each dimension
    enhanced_info = dataset_info.copy()
    enhanced_info['downsample_options'] = {}
    
    for dim, size in dataset_info['dimension_sizes'].items():
        enhanced_info['downsample_options'][dim] = get_downsample_options(size)
    
    return enhanced_info

@app.get("/data-slice")
async def get_data_slice(
    data_var: str,
    axis1: str,
    axis2: str,
    other_indices: str = "{}",
    downsample_factor: int = Query(1, ge=1, le=16, description="Downsampling factor for performance"),
    auto_downsample: bool = Query(False, description="Automatically calculate optimal downsampling")
):
    """Get a 2D slice of the data for visualization with optional downsampling"""
    if dataset_worker is None:
        return {"error": "No dataset loaded"}
    
    try:
        other_idx = json.loads(other_indices)
        
        # Create selection dictionary
        selection = {}
        display_shape = []
        
        for dim in dataset_info['dimensions']:
            if dim == axis1:
                selection[dim] = slice(None)
                display_shape.append(dataset_info['dimension_sizes'][dim])
            elif dim == axis2:
                selection[dim] = slice(None)
                display_shape.append(dataset_info['dimension_sizes'][dim])
            elif dim in other_idx:
                selection[dim] = other_idx[dim]
            else:
                selection[dim] = 0
        
        # Auto-calculate downsampling if requested
        if auto_downsample:
            downsample_factor = calculate_optimal_downsample(tuple(display_shape))
            logger.info(f"Auto-calculated downsample factor: {downsample_factor}")
        
        # Get the data slice using Ray
        data_array = ray.get(dataset_worker.get_data_slice.remote(
            data_var, selection, downsample_factor
        ))
        
        # Get coordinate values
        coords = ray.get(dataset_worker.get_coordinates.remote(
            [axis1, axis2], downsample_factor
        ))
        
        return {
            "data": data_array.tolist(),
            "coord1": coords.get(axis1, []),
            "coord2": coords.get(axis2, []),
            "axis1": axis1,
            "axis2": axis2,
            "shape": data_array.shape,
            "min_val": float(np.min(data_array)),
            "max_val": float(np.max(data_array)),
            "downsample_factor": downsample_factor,
            "original_shape": display_shape
        }
        
    except Exception as e:
        logger.error(f"Error in get_data_slice: {e}")
        return {"error": f"Error getting data slice: {str(e)}"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message["type"] == "animate":
                await animate_data(websocket, message["params"])
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)

async def animate_data(websocket: WebSocket, params: Dict[str, Any]):
    """Stream animation frames via WebSocket using Ray for parallel processing"""
    if dataset_worker is None:
        await manager.send_personal_message(
            json.dumps({"error": "No dataset loaded"}), websocket
        )
        return
    
    try:
        data_var = params["data_var"]
        axis1 = params["axis1"]
        axis2 = params["axis2"]
        animate_axis = params["animate_axis"]
        other_indices = params.get("other_indices", {})
        fps = params.get("fps", 10)
        downsample_factor = params.get("downsample_factor", 1)
        batch_size = params.get("batch_size", 4)  # Process frames in batches
        
        # Create base selection
        selection = {}
        for dim in dataset_info['dimensions']:
            if dim == axis1:
                selection[dim] = slice(None)
            elif dim == axis2:
                selection[dim] = slice(None)
            elif dim == animate_axis:
                selection[dim] = 0  # Will be updated per frame
            elif dim in other_indices:
                selection[dim] = other_indices[dim]
            else:
                selection[dim] = 0
        
        # Get animation axis size
        axis_size = dataset_info['dimension_sizes'][animate_axis]
        total_frames = axis_size // downsample_factor if downsample_factor > 1 else axis_size
        
        # Send animation start message
        await manager.send_personal_message(
            json.dumps({
                "type": "animation_start",
                "total_frames": total_frames,
                "downsample_factor": downsample_factor
            }), 
            websocket
        )
        
        # Process frames in batches for better performance
        for batch_start in range(0, axis_size, batch_size):
            batch_end = min(batch_start + batch_size, axis_size)
            frame_indices = list(range(batch_start, batch_end, downsample_factor))
            
            # Process batch in parallel using Ray
            frame_futures = []
            for frame_idx in frame_indices:
                future = process_animation_frame.remote(
                    dataset_worker, data_var, selection, animate_axis, 
                    frame_idx, downsample_factor
                )
                frame_futures.append((frame_idx, future))
            
            # Get results and send frames
            for frame_idx, future in frame_futures:
                try:
                    frame_data = ray.get(future)
                    if "error" in frame_data:
                        await manager.send_personal_message(
                            json.dumps(frame_data), websocket
                        )
                        return
                    
                    frame_data["total_frames"] = total_frames
                    await manager.send_personal_message(
                        json.dumps(frame_data), websocket
                    )
                    
                    # Control frame rate
                    await asyncio.sleep(1.0 / fps)
                    
                except Exception as e:
                    logger.error(f"Error processing frame {frame_idx}: {e}")
                    await manager.send_personal_message(
                        json.dumps({"error": f"Frame {frame_idx} error: {str(e)}"}),
                        websocket
                    )
        
        # Send animation complete message
        await manager.send_personal_message(
            json.dumps({"type": "animation_complete"}), websocket
        )
        
    except Exception as e:
        logger.error(f"Animation error: {e}")
        await manager.send_personal_message(
            json.dumps({"error": f"Animation error: {str(e)}"}), websocket
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
