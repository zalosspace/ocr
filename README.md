# Optical Character Recognition

## Image Preprocessing
### Normalization
Changes the range of pixel intensity values, converting to values to a specific
range, usually 0 to 1 or 0 to 255. Might help to make text more distinct 
form the background.

### Resolution
Image resolution can influence both processing time and result accuracy.
Higher resolution can significantly increase processing time, while lower 
resolution can impair solution accuracy.
- Font > 8, 300 DPI
- Font <= 8, 400-600 DPI

### Image Binarization
This process converts color images into black and white images, A widely used
technique is **adaptive binarization**, which uses neighbouring pixels as a basis
for performing this conversion. Unlike **normalization**, binarization converts
pixel into two value, usually 0 & 1, or 0 & 255

### Contrast and Sharpness
Increasing the contrast between the text and the background facilitates
the distinction of the character.
The use of local contrast is more benificial than global contrast, as different
parts of the image may have distinct contrast. A technique used for this is 
**Contrast Limited Adaptive Histogram Equalization (CLAHE)**

### Image Geometric Transformation
The way image is captured, has a strong relationship with the different types 
of misalignments of an image
- **Orientation:** We might need to rotate the image for the right orientation
- **Tilt:** We need to detect and adjust the tilt angle so that the text does 
not become inclined
- **Trapezoidal Distortion:** When capturing with cameras, its likely that the
image will resemble a trapezoid instead of a rectangle. To correct this, first
detect the trapezoid, transform it into a rectangle by removing the edges with
no information
- **Depth Perspective:** The image might have varying font size from the top to 
the bottom due to angle of capture
- **Curved Lines:** Self explanatory

### Denoising
Scanned or photographed images can contain random pixel, compression artifacts,
sensor noise, etc.
It might remove period, comma, apostrophes if not used correctly. Here are some
commonly used techniques: 
- **Gaussian Blur:** Smooths high-frequency noise
- **Median Blur:** particularly useful for salt and pepper noise
- **Bilateral Filter:** Reduces noise while preserving edges
