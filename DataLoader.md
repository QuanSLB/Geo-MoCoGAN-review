* RawDataFile
    * facies
        * channel_lag, crevasse_splay, erosion, hiatus, levee, mud_pug, overbank, point_bar
    * properties
        * thickness, topograpghy, facies-calssficationn, grain-szie, porosity, etc
>  RawDataFile is encrypted by a hash:
> 1. Class: 3d_models
> 1. Filename: xxxxx.h5
> 1. most recent modified time

_**RawDataFile is encrypted by a hash with 3 properties:**_

> - Class: 3d_models
> - Filename: xxxxx.h5
> - most recent modified time

> e.g., hash of  **RawDataFile: video_i**
```python
class = "3d_models"
filename = "ChannelBelt3D-1dad52d4-94b3-3e25-af1a-6f6e05a8c2d2-25865-566cbe8e-3b6a-3cb8-8fbd-b2626e206d25-36997.h5"
mtime = os.path.getmtime(filename)

hash_str = "[('3d_models', 'ChannelBelt3D-1dad52d4-94b3-3e25-af1a-6f6e05a8c2d2-25865-566cbe8e-3b6a-3cb8-8fbd-b2626e206d25-36997.h5', 1651766445.0)]"

hash = hashlib.md5(hash_str.encode('utf-8')).hexdigest()
```

To load `RawDataFile` to `ImageDataset` or `VideoDataset`


```python
    def ressim_3dmodel_h5_loader(Str): 
        # input is path-to-RawDataFile
        # if CacheDataFile exists, it will load CacheDataFile and output Video Numpy Array List
        # if CacheDataFile does not exist, it will read RawDataFile and create DictList
     
    def _create_numpy_array_from_dictlist(DictList):
        # select 9 channels: 8 facies + 1 thickness from DictList keys
        # generate video numpy array in shape: 256 x {256x129} x 9
        # the output should be video_original_numpy
```
```
    Video Dict List{
        - List contains videos
        - each video is a dict
        - dict contains keys selected from RawDataFile: ntg, fc, gs, etc
        - dict value is a 3D numpy array of shape 129 x 256 x 256
        - first dimension 129 is video length, 256 x 256 is domain size
     }
     
    Video Numpy Array List{
        - List contains videos
        - each video is a 3D numpy array of shape 256 x 33024 x 9
        - last dimension is channel: 1 thickness + 8 facies
        - second dimension is concatnated: 33024 = 129 x 256
        - 129 is video length, 256 is domain size
     }
```
    
        
```mermaid
graph LR;
    subgraph Input
        A[RawDataFile] 
    end
    A-->|check cache<br>loader.get_cache_path|B
    subgraph VideoFolder._build_cache
        direction LR
        subgraph loader.ressim_3dmodel_h5_loader
            direction TB
            B{cache exists?}
            B -->|Yes| C{{Load CacheDataFile}}
            B -->|No| AB{{Load RawDataFile}}
            AB -->  ABC((Select: <br>facies_classification<br>net_to_gross<br>grain_size))
            ABC-->H[Video DictList]
        end


        C --> E[Video Numpy Array List<br> List of videos <br>each video: 256x33024x9]
        H --> VideoFolder._create_numpty_from_dictlist 
        %%HI((Select 9 channels:<br> 8 facies <br>+<br>1 thickness))

        subgraph VideoFolder._create_numpty_from_dictlist
            direction TB
            HI{{Select 9 channels:<br> 8 facies +1 thickness}}--> HC{{concatenate: axis=1}}
        end
        VideoFolder._create_numpty_from_dictlist-->E
        
    end

    E-.->|write cache file|G[CacheDataFile]
    E --> ID[ImageDataset<br>frame, label]
    E --> VD[VideoDataset<br>clip, label]

    subgraph Outputs
        ID
        VD
    end
```



```mermaid
---
Data Loader
---
classDiagram 
    class VideoFolder{
            root_dir: './datasets/ressim-256-cells/3d_models'
            ext: '.h5'
            tag: 'vae'
            
            _build_hash(): create hash name of RawDataFile
            _build_cache(): write CacheDataFile of numpy array
            __len__(): number of videos
            __getitem__(id): read entire video from CacheDataFile
           _create_numpy_array_from_dictlist(DictList)
    } 
    class RawDataFile{
      -name : file_name.h5
      -path : './datasets/ressim-256-cells/3d_models'
      -entry : timesteps, all-zero or current
      -subentities: facies, properties
    }

    class CacheDataFile{
      -name : file_name.h5
      -path : './datasets/cache/ressim-256-cells-3d_models'
      -entity : original and transform
      -subentities: numpuy array 256x33024x9
    }
    
   RawDataFile --|> CacheDataFile: VideoFolder._build_cache()
```
