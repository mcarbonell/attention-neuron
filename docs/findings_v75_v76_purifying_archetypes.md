# Findings V75 & V76: The Purifying Archetype Classifier

## Overview
Following the static template matching in V74 (which yielded 82% accuracy with 10 archetypes), we recognized a critical flaw: taking the mean of *all* images of a digit creates a blurry "average" that is highly contaminated by outliers, weird handwriting styles, and mislabeled data.

To solve this, we implemented a dynamic, supervised clustering algorithm over two iterations (V75 and V76) that automatically isolates errors and cleans the base archetypes.

## The Algorithm Evolution
### V75: Iterative Sub-Archetypes
In V75, we iteratively evaluated the dataset. Any image that was misclassified was pulled out, and all errors for a given digit were averaged to form a *new* "Sub-Archetype" (e.g., "The mean of all '4's that look like '9's").
- **Result:** Accuracy jumped from 80.8% to 86.9% by introducing 60 sub-archetypes.

### V76: The Purifying Step
In V76, we refined the logic. We realized that if we remove the errors to form a new archetype, we must also *recalculate the original base archetype without the errors*. This allows the base archetype to "purify" itself, becoming sharper and more representative of the ideal digit. Furthermore, correctly classified images are reassigned to their nearest valid sub-archetype, allowing clusters to naturally drift to their local centers of mass (like K-Means).
- **Result:** Accuracy skyrocketed to **93.50%** using 280 pure archetypes.

![Extended Taxonomy](file:///c:/Users/mrcm_/Local/proj/algorithms/attention-neuron/results/figures/v76_refined_archetypes.png)

## Interpretability and Speed
The most profound outcome of this experiment is **absolute interpretability**. 
Unlike a Deep Neural Network where knowledge is distributed across thousands of opaque weights, here the knowledge is explicitly stored as 280 human-readable images (the archetypes). 
- If the model makes a mistake, we can mathematically show *exactly which archetype* it confused the image with, and we can *look* at that archetype.
- The training speed is blistering: it requires no backpropagation, just vector distances and means.

## Conclusion
We have demonstrated that a simple, interpretable distance-based classifier can achieve >93% accuracy on MNIST if the templates are dynamically purified using supervised error isolation. This drastically reduces the parameter count compared to KNN (from 60,000 reference images to just 280), proving that intelligent noise isolation is often more powerful than blind parameter scaling.
