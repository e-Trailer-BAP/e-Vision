
**360 camera system:**

------------------------------------------------------------------------------------------------------------------------------------------

[360 Degree Camera System in 2D & 3D: What You Need to Know | Kocchi's (kocchis.com)](https://www.kocchis.com/blog/360-degree-camera-system-in-2d-3d-what-you-need-to-know-kocchis/ "https://www.kocchis.com/blog/360-degree-camera-system-in-2d-3d-what-you-need-to-know-kocchis/")

-   **Useful for general overview**

------------------------------------------------------------------------------------------------------------------------------------------

[How Does 360 Car Camera Work | Should You Install One? (kocchis.com)](https://www.kocchis.com/blog/how-does-360-car-camera-work-should-you-install/ "https://www.kocchis.com/blog/how-does-360-car-camera-work-should-you-install/")

Speaks about having bird's eye view of the car.

-   Surround-view camera systems
-   Car overhead view systems covering entire are surrounding a vehicle with a 360 video at the top down
    -   Like having own personal drown flying right above your car transmitting live imagery directly to your in-car monitor to help remove blind spots

System:

-   Minimum 4 cameras needed, can be more (8/9)
-   Most of the cameras in such a system would need to have a wide-angle lens up to 180 degrees.
-   4 cameras create images from all measurements, concurrently sent to ECU (Electronic Control Unit), processed for correction, stitched together to generate real-time perspective of the environment of the vehicle as if from above.
-   Show component and stitched together images on display

Available systems:

-   2D and 3D systems available

Display Hardware:

-   Panoramic image processing software shown on an HMI (Human Machine Interface)

Cars with 360-Degree Camera Systems:

-   Audi: Top view camera system with Virtual 360 View
-   Jaguar: 360 Surround Camera
-   BMW: Surround View With 3D View
-   Chevrolet: Surround Vision€
-   Ford: 360-Degree Camera
-   Hyundai: Surround View Monitor
-   Infiniti: Around View Monitor
-   Kia: Surround View Monitor
-   Land Rover: 360-Degree Parking Aid; ClearView
-   Mazda: 360-Degree View Monitor
-   Mercedes-Benz: Surround View System
-   Nissan: Around View Monitor
-   Toyota: Bird’s Eye View Camera
-   Volkswagen: Overhead View Camera (Area View)
-   Volvo: 360-Degree Surround View

------------------------------------------------------------------------------------------------------------------------------------------

[CVPR 2014 Open Access Repository (cv-foundation.org)](https://www.cv-foundation.org/openaccess/content_cvpr_workshops_2014/W17/html/Zhang_A_Surround_View_2014_CVPR_paper.html "https://www.cv-foundation.org/openaccess/content_cvpr_workshops_2014/w17/html/zhang_a_surround_view_2014_cvpr_paper.html")

[A Surround View Camera Solution for Embedded Systems (cv-foundation.org)](https://www.cv-foundation.org/openaccess/content_cvpr_workshops_2014/W17/papers/Zhang_A_Surround_View_2014_CVPR_paper.pdf "https://www.cv-foundation.org/openaccess/content_cvpr_workshops_2014/w17/papers/zhang_a_surround_view_2014_cvpr_paper.pdf")

[Surround view camera systems for ADAS (Rev. A) (ti.com)](https://www.ti.com/lit/wp/spry270a/spry270a.pdf?ts=1713863317774&ref_url=https%253A%252F%252Fwww.google.com%252F "https://www.ti.com/lit/wp/spry270a/spry270a.pdf?ts=1713863317774&ref_url=https%253a%252f%252fwww.google.com%252f")

**Abstract:**

Automotive surround view camera system is an emerging automotive ADAS (Advanced Driver Assistance System) technology that assists the driver in parking the vehicle safely by allowing him/her to see a top-down view of the 360◦ surroundings of the vehicle. Such a system normally consists of four to six wide-angle (fish-eye lens) cameras mounted around the vehicle, each facing a different direction. From these camera inputs, a composite bird-eye view of the vehicle is synthesized and shown to the driver in real-time during parking. In this paper, we present a surround view camera solution that consists of three key algorithm components: geometric alignment, photometric alignment, and composite view synthesis. Our solution produces a seamlessly stitched bird-eye view of the vehicle from four cameras. It runs realtime on DSP C66x producing an 880 × 1080 output video at 30 fps. Keywords-surround view cameras; around view camera systems; multi-camera fusion; ADAS; geometric alignment; photometric alignment; composite view synthesis;

**NOTES:**

**General purpose source:**

This source focuses on building a top-down view of the 360 degree surroundings of the vehicle. The design specifically uses four fish-eye cameras with 180 degree Field Of View (FOV) which are mounted around the vehicle, each facing in a different direction. The solution is implemented on a toy jeep to illustrate the results. The camera solution produces a seamlessly stitched bird-eye view, designed specifically for embedded systems. It is implemented on DSP C66x and produces high definition output video at 30fps. By relying of specially designed calibration charts, the calibration of the system is made efficient.

**Specific techniques used:**

The algorithm for the proposed surround view solution has three components.

The first component is geometric alignment or calibration. This refers to the correcting of fish eye distortion and the conversion of each input frame from the fish eye to a common bird-eye perspective.

-   Algorithm has calibration-chart-based approach. During the calibration of the cameras, four specially designed calibration charts, designed to reliably find matching features, are placed in common FOV's of adjacent cameras. A frame is then captured from each camera.
-   Fish-eye distortion correction is then implemented by applying the inverse transformation of the radial distortion function to each frame.
-   After this Lens Distortion Correction (LDC) has been performed, camera-specific transformation matrices are estimated and applied to the corrected frames, in order to register all four input views with the ground plane.
    -   The initial perspective transformation is done with parameters estimated from the frame content itself.
    -   Next, a combination of Harris corner detection and calculations of BRIEF scores is used to find optimal perspective matrix for each frame.
    -   This information (on both the LDC and the perspective transformation) is stored in a geometric look-up-table, which will be used during the composite view synthesis to create the final surround view output.

The second component is photometric alignment. Due to different scene illumination, the color and brightness of the same object captured by different cameras (in overlapping regions of view) can be quite different. These differences are noticeable in a stitched composite image. Photometric alignment corrects the brightness and color mismatches between adjacent views to achieve seamless stitching.

-   For this, color and brightness correction functions are designed for each camera view.
    -   Specifically, tone mapping functions for each RGB channel of each input camera are estimated by adjusting them to minimize the total mean square error of pixel value discrepancies in overlapping regions.
        -   This is used since at each location in overlapping-view regions, there are two pixels available (from the different adjacent views) for a single physical point in space. The pixel value discrepancy is then defined as the difference between the pixel values of these pixels (captured by different cameras).
    -   The tone mapping functions are jointly optimized for all cameras for each color channel. This process is then repeated independently for the different color channels.
    -   Applying these function to the input frames then achieves photometric correction.

The last component is the synthesis algorithm, which actually generates the composite surround view after the geometric and photometric corrections have been performed.

-   Synthesis creates the stitched output image using the mapping for the fish-eye input streams available in the geometric LUT.
    -   Output in non-overlapping region: single pixel fetched from geometric LUT, described photometric alignment performed to input pixel to get final output pixel value.
    -   Output in overlapping region: two pixels are fetched from geometric LUT (as image data from two adjacent input frames matches output pixel). Solution uses photometric alignment together with specific blending techniques to generate optimal output pixels.

The paper provides an overview of how these three components interact within the total architecture, as well as ways to optimize this architecture for the specific DSP on which it is implemented to allow for real-time video frame rates for high-resolution output images.

------------------------------------------------------------------------------------------------------------------------------------------

[(PDF) Surround Video: A Multihead Camera Approach (researchgate.net)](https://www.researchgate.net/publication/220067625_Surround_Video_A_Multihead_Camera_Approach "https://www.researchgate.net/publication/220067625_surround_video_a_multihead_camera_approach")

**Surround video: a multihead camera approach**

**Frank Nielsen**

**Sony Computer Science Laboratories, Tokyo, Japan**

**E-mail: Frank.Nielsen@acm.org**

**Published online: 3 February 2005**

**Springer-Verlag 2005**

**Abstract:**

We describe algorithms for creating, storing and viewing high-resolution immersive surround videos. Given a set of unit cameras designed to be almost aligned at a common nodal point, we ﬁrst present a versatile process for stitching seamlessly synchronized streams of videos into a single surround video corresponding to the video of the multihead camera. We devise a general registration process onto raymaps based on minimizing a tailored objective function. We review and introduce new raymaps with good sampling properties. We then give implementation details on the surround video viewer and present experimental results on both real-

world acquired and computer-graphics rendered full surround videos. We conclude by mentioning potential applications and discuss ongoing related activities.

Video supplements:

[http://www.csl.sony.co.jp/person/nielsen](http://www.csl.sony.co.jp/person/nielsen "http://www.csl.sony.co.jp/person/nielsen")

Key words: Virtual reality – stitching – environment mapping

------------------------------------------------------------------------------------------------------------------------------------------

[Sci-Hub | | 10.1007/978-3-540-78157-8_16](https://sci-hub.se/10.1007/978-3-540-78157-8_16 "https://sci-hub.se/10.1007/978-3-540-78157-8_16")

**Bird’s-Eye View Vision System for Vehicle Surrounding Monitoring**

**Yu-Chih Liu, Kai-Ying Lin, and Yong-Sheng Chen**

**Department of Computer Science, National Chiao Tung University, Hsinchu, Taiwan**

**Abstract:** Blind spots usually lead to difficulties for drivers to maneuver their vehicles in complicated environments, such as garages, parking spaces, or narrow alleys. This paper presents a vision system which can assist drivers by providing the panoramic image of vehicle surroundings in a bird’s-eye view. In the proposed system, there are six fisheye cameras mounted around a vehicle so that their views cover the whole surrounding area. Parameters of these fisheye cameras were calibrated beforehand so that the captured images can be dewarped into perspective views for integration. Instead of error-prone stereo matching, overlapping regions of adjacent views are stitched together by aligning along a seam with dynamic programming method followed by propagating the deformation field of alignment with Wendland functions. In this way the six fisheye images can be integrated into a single, panoramic, and seamless one from a look-down viewpoint. Our experiments clearly demonstrate the effectiveness of the proposed image-stitching method for providing the bird’s eye view vision for vehicle surrounding monitoring.

------------------------------------------------------------------------------------------------------------------------------------------

[Sci-Hub | A Vision Based Top-View Transformation Model for a Vehicle Parking Assistant. Sensors, 12(4), 4431–4446 | 10.3390/s120404431](https://sci-hub.se/10.3390/s120404431 "https://sci-hub.se/10.3390/s120404431")

**A Vision Based Top-View Transformation Model for a Vehicle Parking Assistant**

**Chien-Chuan Lin * and Ming-Shi Wang**

**Department of Engineering Science, National Cheng Kung University Taiwan, No.1, University Road, Tainan City 701, Taiwan; E-Mail: mswang@mail.ncku.edu.tw**

**Abstract:** This paper proposes the Top-View Transformation Model for image coordinate transformation, which involves transforming a perspective projection image into its corresponding bird’s eye vision. A fitting parameters searching algorithm estimates the parameters that are used to transform the coordinates from the source image. Using this approach, it is not necessary to provide any interior and exterior orientation parameters of the camera. The designed car parking assistant system can be installed at the rear end of the car, providing the driver with a clearer image of the area behind the car. The processing time can be reduced by storing and using the transformation matrix estimated from the first image frame for a sequence of video images. The transformation matrix can be stored as the Matrix Mapping Table, and loaded into the embedded platform to perform the transformation. Experimental results show that the proposed approaches can provide a clearer and more accurate bird’s eye view to the vehicle driver.

**Keywords:** top-view transformation; bird’s eye view; inverse perspective mapping

asdfladjsdddf

--------------------------------------------------------------------------------------------------------------------------------------------

[Sci-Hub | 360° Surround View System with Parking Guidance. SAE International Journal of Commercial Vehicles, 7(1), 19–24 | 10.4271/2014-01-0157](https://sci-hub.se/10.4271/2014-01-0157 "https://sci-hub.se/10.4271/2014-01-0157")

**360° Surround View System with Parking Guidance**

**Mengmeng Yu and Guanglin Ma**

**Delphi Automotive**

**Abstract:** In this paper, we present a real-time 360 degree surround system with parking aid feature, which is a very convenient parking and blind spot aid system. In the proposed system, there are four fisheye cameras mounted around a vehicle to cover the whole surrounding area. After correcting the distortion of four fisheye images and registering all images on a planar surface, a flexible stitching method was developed to smooth the seam of adjacent images away to generate a high-quality result. In the post-process step, a unique brightness balance algorithm was proposed to compensate the exposure difference as the images are not captured with the same exposure condition. In addition, a unique parking guidance feature is applied on the surround view scene by utilizing steering wheel angle information as well as vehicle speed information.

**NOTES:**

**General purpose source:**

This source proposes a solution for a real-time 360 degree surround system. The solution uses four 180 degree fish-eye cameras mounted to cover the entire surrounding area. The work has a real-time running and low cost requirement, which is reflected in the design choices for efficient color and brightness compensation and the flexible stitching method. The paper also provides an extension of the system by describing a parking guidance feature.

**Specific techniques used:**

The proposed algorithm follows a similar general structure as other surround view system solutions.

The first step in the algorithm consists of correcting the radial distortion introduced by the used wide-angle fish-eye cameras. This distortion is corrected using distortion curves which are provided by the supplier of the used lenses. The information for the distortion correction is encoded in a look-up-table (LUT). Using this correction technique, the resulting images will still have some imperfections/"ripples". In order to remove these imperfections, a 2D interpolation look-up table is created. The interpolation is performed based on the four pixels surrounding the pixel of interest in order to find the sought-for intensity.

The second step in the algorithm consists of projecting the corrected images into the top-down view plane. A projection matrix is established by specifying feature points in a reference image of the ground plane together with their corresponding points in the captured images. By doing this, a top-down view projection look-up table is obtained. The look-up tables for the distortion correction and for the projection are combined into a single table to reduce computational cost.

The next step in the algorithm consists of arranging the four corrected and projected images captured by the different cameras in their corresponding places, and stitching them into one top-down view image. The proposed solution focuses on simplicity, as the opted for fusion method is based on using a pixel value weighted average. Output pixel values that only have one corresponding input pixel (from a single camera) map directly to said input pixel. In the overlapping regions of view of the cameras the mentioned weighted average is used, based on a computed weight factor. This weight factor is dependent on the number of pixels in the overlap region of adjacent images.

After the stitching has been achieved, color and brightness compensation are implemented to perfect the result. This is necessary as each of the stitched-together images is captured at different conditions, resulting in intensity differences. The proposed solution uses a gain compensation algorithm, where parameters are calculated by comparing the pixel values captured by different cameras for overlapping pixels.

**--------------------------------------------------------------------------------------------------------------------------------------------**

[Sci-Hub | | 10.1109/tits.2017.2750087](https://sci-hub.se/10.1109/tits.2017.2750087 "https://sci-hub.se/10.1109/tits.2017.2750087")

**3-D Surround View for Advanced Driver Assistance Systems**

**Yi Gao, Chunyu Lin, Yao Zhao, Senior Member, IEEE, Xin Wang, Shikui Wei, and Qi Huang**

**Abstract:** As the primary means of transportations in modern society, the automobile is developing toward the trend of intelligence, automation, and comfort. In this paper, we propose a more immersive 3-D surround view covering the automobiles around for advanced driver assistance systems. The 3-D surround view helps drivers to become aware of the driving environment and eliminates visual blind spots. The system first uses four fish-eye lenses mounted around a vehicle to capture images. Then, according to the pattern of image acquisition, camera calibration, image stitching, and scene generation, the 3-D surround driving environment is created. To achieve the real-time and easy to-handle performance, we only use one image to finish the camera calibration through a special designed checkerboard. Furthermore, in the process of image stitching, a 3-D ship model is built to be the supporter, where texture mapping and image fusion algorithms are utilized to preserve the real texture information. The algorithms used in this system can reduce the computational complexity and improve the stitching efficiency. The fidelity of the surround view is also improved, thereby optimizing the immersion experience of the system under the premise of preserving the information of the surroundings.

**Index Terms:** Fish-eye lens, camera calibration, 3D surround view, image stitching, driver assistance systems.

**Notes:**

**General purpose source:**

The paper proposes a 3D surround view system aiming to eliminate blind spots. The system utilizes four fish-eye lenses mounted around the vehicle to capture images, which are then processed through camera calibration, coordinate transformation, texture mapping, and image fusion to create a comprehensive 3D surround view displayed on the vehicle's central control panel. This paper thus goes beyond having a single 2D perspective from above the vehicle, which it states can be misleading to drivers.

**Specific techniques used:**

While the general structure of this solution follows the correcting of fish-eye images, transformation of view, stitching and brightness/color compensation in producing a seamless surround view of the vehicle, the 3D nature of the solution does add a level of complexity to these steps.

The first step in the solution is camera calibration. For this, a special fish-eye lens calibration algorithm is introduced, utilizing a collinear constraint and edge corner points. This process is facilitated with the use of a checkerboard. The algorithm allows for accurate calibration with minimal human intervention, simplifying the process for installation in vehicles.

The next step is coordinate transformation. The 2D images captured by the fish-eye lenses are transformed into a 3D ship model using a virtual imaging surface. This transformation enables the mapping of image pixels onto the 3D model, facilitating a realistic representation of the vehicle's surroundings.

A 3D ship model is then constructed to serve as the framework for the surround view. This model provides a consistent and immersive visualization of the vehicle's environment. Texture mapping is employed to enhance the realism of the 3D surround view by mapping image textures onto the surfaces of the 3D model. This process ensures that the generated surround image is natural. Lastly, image fusion is necessary. Overlapping regions of the surround view images are fused together using alpha fusion techniques. This produces seamless transitions between adjacent images and eliminates brightness differences.

**-------------------------------------------------------------------------------------------------------------------------------------------**