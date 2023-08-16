# Overview
This is the source code corresponding to the paper with the title “Data-Guided Authoring of Procedural Models of Shapes”, published at Pacific Graphics 2023. The main idea is to provide the author of a procedural model with an assistive tool that allows for iterative refinement of the procedural model being developed. Given a collection of reference shapes, at each iteration, the automatic component trains a model to identify reference shapes that can be replicated by the current procedural model, as well as shapes that cannot be replicated (incompatible shapes). In addition, it groups similar looking incompatible shapes together, so that the author can quickly inspect them and decide how the procedural model can be further improved.

We demonstrate the concept by starting with a basic procedural model for creating 3D shapes of tables and incrementally improving it in a number of iterations. There are a total of 5 versions of the procedural model, each an improvement over the last one.

# Package requirements
The project is implemented using Python 3.10 and depends on the following python packages.
```
numpy
trimesh
tqdm
scikit-learn
torchvision
pytorch3d
zlib
bpy
opencv-python
mathutils
```

# The Procedural Model
You can see examples of the created shapes by each of the different versions of the procedural model by running the following command. You can use any version number between 1 and 5.

```python procedure.py --version=5 --num_samples=5```

# Training Data Generation
For each iteration, the training data is composed by randomly sampling the parameter vector and splitting the dataset into training data and test data. We already include the data in the ```data``` directory. However, if you are interested in modifying the procedural models, fresh datasets can be created using the following command.

```python dataset.py --version=5```

# Training/Testing
To train a model for a particular version of the procedural model, use the following command.

```python train.py --version=5 --split_ratio=0.9 --batch_size=32 --num_epochs=50```

Alternatively, you can download the trained models from https://drive.google.com/drive/folders/1ID0PIPA3CgwH6VD845rQH-GxRKLxkgwo?usp=sharing

To test the performance of a model, use the following command.

```python test.py --version=5 --batch_size=32```

# Replicating Unseen Shapes
We use shapes from the ShapeNet collection of 3D shapes to demonstrate how the method identifies compatible and incompatible shapes from an unseen collection of shapes. The list of ShapeNet shapes is in ```shapenet.csv```. To see how any particular version is able to replicates the shapes, run the following command.

```python replicate.py --version=5```

The result is stored in an image named ```replicated_{version_number}.png```. Similar looking shapes are color coded with the same color and grouped together. Good replications are colored green and other replications are colored yellow.


# Acknowledgement
The procedural model used in the main text of the paper is authored by Sharjeel Ali and the source code is hosted at https://github.com/SharjeelAliCS/Procedural-Modeling-Library.git

Original implementation of the Light Field Descriptors used in this study can be found at https://github.com/Sunwinds/ShapeDescriptor.git

We also modified the Python wrapper for the original LFD available at https://pypi.org/project/light-field-distance/
