# Birds Eye View Calibration Toolkit

The Inverse Perspective Mapping (IPM), is the process of converting a perspective image to a perpendicular top-to-bottom view image, also known as Bird's Eye View Mapping (BEW).
This process involves some initial calibration steps. The toolkit in this repository provides a calibration technique using Python and OpenCV which is applicable to both manual and satellite-image-based calibration methods.

**The tutorial for using the toolkit is outlined in the file Guide.pdf.**

## Requirements
- OS: Windows / Linux / Mac
- Python: 3.8.1 (or above)
- OpenCV: 4.7.0 (or above)
- Numpy: 1.23.5 (or above)

## Calibration Methods
### Manual Calibration ('Calib_GrndPlane.py')
This script allows manual determination of the ground plane and estimation of BEV calibration points from a video file, as follows:
- Background Extraction: Removing the moving objects in the scene.
- ROI Determination: Selecting the region of interest.
- Ground Plane Selection: Marking four points to create a square in the scene.
- Refining Aspect Ratio: Determining pixel-to-meter ratio in two directions.
The process generates a folder with configuration files and images representing each step.

### Satellite-based Calibration ('Calib_SatFeature.py')
This script requires a perpendicular satellite image of the location where the video is recorded and involves the following steps:
- Background Extraction: Removing moving objects in the scene.
- ROI Determination: Selecting the region of interest.
- Point Identification: Selecting at least four points in the satellite image and reidentifying them in the video scene.
- Refining Aspect Ratio: Determining pixel-to-meter ratio in two directions.
Similar to the manual calibration, this process generates a folder with configuration files and step-by-step images.


## References

- Rezaei, M., Azarmi, M., & Mir, F.M.P. (2023). "3D-Net: Monocular 3D object recognition for traffic monitoring." *Expert Systems with Applications*, 227, p.120253. 
 [Paper](https://doi.org/10.1016/j.eswa.2023.120253) | [Code](https://codeocean.com/capsule/7713588/tree/v1) | [Demo](https://www.youtube.com/watch?v=FdiQ_EGbZe0) | [Code Description](https://www.youtube.com/watch?v=XT8izWwNdZo&t=7s)

- Rezaei, M., & Azarmi, M. (2020). "Deepsocial: Social distancing monitoring and infection risk assessment in COVID-19 pandemic." *Applied Sciences*, 10(21), p.7514. 
 [Paper](https://doi.org/10.3390/app10217514) | [Code](https://github.com/DrMahdiRezaei/DeepSOCIAL) | [Demo](https://www.youtube.com/watch?v=FwCP2ySDshE&t=19s&pp=ygUKZGVlcHNvY2lhbA%3D%3D)

