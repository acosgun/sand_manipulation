In main, depending on Step 

---------
trackTool
---------

- all sequences of images that are encoded in images/push.txt or images/tap.txt, are scrolled to track the blue blob corresponding to the tool. 
- Examples of push.txt and tap.txt are in ./share/
- IMPORTANT: make sure the tool is visible in all images for push, particularly this important for first and last image

---> Outputs five files: 
- centroidFileRaw.txt and centroidFileProcessedPush.txt: [userID img centX centY centVX centVY velChange visible] 
where: velChange={0,1} indicates start or end of sequence (vel change) and visible={0,1} indicates wether tool was visible or not
- motPrimFilePush.txt: breaks the sequence into smaller ones, one per push: [userID startImg endImg]
- U_tap_images.txt: [userID img_ini img_fin]
- Y_tap.txt: [centX centY]  


-----------------------------------
generateAndVisualizeDataForLearning 
-----------------------------------

- the motion primitive sequences of images that are encoded in motPrimFilePush.txt, are scrolled to track and sample the sand contour 
- Outputs file toolAndContour.txt: [userID img centX centY centVX centVY velChange visible X1 Y1 ... Xn Yn] the Xi Yi are samples of the contour in that image
- toolAndContour.txt is loaded to get pairs of images with: sufficient distance between contours, sufficient distance between tools and with tool moving left

---> Outputs three files: 
- U_push_images: [userID img_ini img_fin]
- U_push_contours: [X_ini,1 Y_ini,1 ... X_ini,n Y_ini,n X_fin,1 Y_fin,1 ... X_fin,n Y_fin,n]
- Y_push: [centX_ini centY_ini centX_fin centY_fin]  
images of this are shown to check data

//TODO show tap data
