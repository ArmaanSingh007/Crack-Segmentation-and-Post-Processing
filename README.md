Goal: To create pixelwise masks to segment the cracks in the images, focusing on unsupervised methods like clustering (e.g., k-means) or Dbscan. The goal is to identify the exact location and shape of cracks without relying on labeled data for training.
      Develop a strategy to refine the segmentation results and clean up potential errors. This may involve post-processing steps like morphological operations, connected component analysis, or filtering based on size or shape. 
      Evaluate the segmentation results on a small set of 20 images that you manually label. The goal is to assess the accuracy and quality of the segmentation process by comparing the automated results against the ground truth labels you create.

Important note: We use LABELME to label the image .

Conclusion: we do segmentation of unsupervised data by Two clustering method K-means and DBSCAN in which DBSCAN is the best choice among them. K-means clustering is quick and simple but struggles with irregular crack shapes and noise whereas DBSCAN More robust, handles noise better, and works well with complex shapes. Our recommendation is ' DBSCAN is generally more reliable for crack detection in complex, noisy environments and it requires less post-processing than K-means' .





 




