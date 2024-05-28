
# image analysis pipeline for STOMATA


1) Annotate images using an annotation platform. Make Sense AI was used here (https://www.makesense.ai/), 
      - annotate each object in every image using the rectangle tool, making sure to label them appropriately
      - after annotations have been made around every object in the images, download them as VOC XML formatted files
      - save jpegs and xmls into two separate and non-overlapping sets (/Train and /Test)

2) Train the model (model_trainer.py)
      - change save locations
      - set desired batch size for training (2 was used here as a default) and number of epochs to run 
            - 25 epochs were used here, probably overkill- scores leveled off after 10, so run for maybe 15
            - epochs will determine the number of iterations over training and testing datasets that are performed
     - OUTPUTS:
            - model (~165mb file ending in .pth)
            - loss graph (essentially tells you the model error rate as it's learning)

3) Run the model on new imageset (image_processor.py)
      - change save locations and model names as necessary
      - OUTPUTS:
            - /processed/xy_final.csv (contains information about boxes applied on each image)
                    - contains a, b, and c^2 counts IN PIXELS as well as the id of the image they came from 
            - /processed/images.jpeg (all processed images with appropriate boxes surrounding them)
      - TO CONVERT pixels to mm, use a conversion factor of 1310px:1mm for 40x images

4) Check the model reflects real counts (model_check.R)
      - reads in python csv's produced earlier, combines them with prior runs (cbind) and computes an lm to test machine counted ~ hand counted
      - change save locations and df names as necessary (and thresh)
      - OUTPUTS:
            - makes a table of counts for each image (test_model.csv)
            - calculates a linear model and makes a ggplot to estimate residual variance and R^2
            - there's also a pointless section that calculates a spline regression of r^2 across a sample of thresholds to figure out the threshold that maximizes r^2 (it was wrong)


5) Supplemental scripts
      - threshold images to make features more distinct (image_threhsolder.py)
      - convert images to desired format (tiff_to_jpg.py)
