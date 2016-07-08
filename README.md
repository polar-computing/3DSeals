# 3DSeals
Assessing the movements and distribution of ice seals in a warming Antarctic; developing tools to automate the detection of seals in imagery, to identify individual seals based on unique body patterning, and to infer 3-dimensional seals from 2D images.

# Decription
Weddell seals are ice seals, meaning they live on, among, and under the pack ice of Antarctica year-round. Most Weddell research has taken place in areas of heavy ice cover, with little work on the Antactic Peninsula, an area rich in marine life. This project aims to understand the distribution of Weddell seals on the Peninsula and their movements within and among years by tracking them with opportunisitic photography sourced from researchers and tourists. We would like to automate this process as much as reasonable, as manual photo-identification of animals is a very time-intensive task. 

If time allows, a 3D element would be an interesting ID solution, whereby tourist photos often shot at "artistic" angles, and those shot at the correct angle, could be draped over a wireframe of a seal, with future photos filling in the gaps. This could involve identifiers for body parts (face, fore flippers, rear flippers, etc.) whose angles could be used to predict the 3D orientation of the seal from 2D images.

#Challenges
Ideal images are taken perpendicular to the seal when it is lying on its side with its ventral region (our photo-identification region) facing the camera.

Angle: 
Variation in seal yaw and roll relative to the camera is common as are contortions of the body axis.

The Roly-Poly factor:
Weddell seals have a thick blubber layer, which causes rolls in the skin, distorting patterning.

Fur condition:
An annual molt coincides roughly with the time of year when photographers are in the Antarctic and causes the patterning on the fur to become less distinct. Similarly, the patterning on wet seals is more distinct than on dry seals. 

Image quality:
Images are submitted by researchers and members of the public with a wide range of photographic equipment and skill, leading to variability in quality. 


# Data
## Input data:
Cropped jpeg files of Weddell Seals. 

File size: 33.2MB zipped HERE--
--34.2MB unzipped, no of files HERE--

Cropped jpeg files of Weddell seals organized into groups of matched images and all other unique individuals -as assessed by manual photo-ID


# Tools
Python, Matlab, NumPy, CNNs

We have had some limited success using [Hotspotter](https://github.com/Erotemic/hotspotter) for pattern recognition. It is being replaced by [IBEIS](https://github.com/Erotemic/ibeis) which is in development.

This project has similarities to the [Right Whale Recognition Kaggle Challenge](http://deepsense.io/deep-learning-right-whale-recognition-kaggle/). Source code and trained models available [here](https://www.dropbox.com/s/rohrc1btslxwxzr/deepsense-whales.zip?dl=1).


