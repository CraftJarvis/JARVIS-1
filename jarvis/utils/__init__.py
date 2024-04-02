
import numpy as np
from typing import List, Union

def write_video(
    file_name: str, 
    frames: Union[List[np.ndarray], bytes], 
    width: int = 640, 
    height: int = 360, 
    fps: int = 20
) -> None:
    import av
    """Write video frames to video files. """
    with av.open(file_name, mode="w", format='mp4') as container:
        stream = container.add_stream("h264", rate=fps)
        stream.width = width
        stream.height = height
        for frame in frames:
            frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
            for packet in stream.encode(frame):
                    container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)