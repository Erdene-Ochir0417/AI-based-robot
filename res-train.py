import os
import keras
import tensorflow as tf
from keras_retinanet import models
from keras_retinanet import layers
from keras_retinanet import losses
from keras_retinanet.models.retinanet import retinanet_bbox
from keras_retinanet.preprocessing.csv_generator import CSVGenerator
from keras_retinanet.utils.anchors import make_shapes_callback
from keras_retinanet.utils.model import freeze as freeze_model
from keras_retinanet.callbacks import RedirectModel
from keras_retinanet.callbacks.eval import Evaluate
import csv

with open("an.csv") as text1:
	uploaded1 = text1.read()
	
with open("class.csv") as text2:
	uploaded2 = text2.read()
	
with open("validation.csv") as text3:
	uploaded3 = text3.read()


backbone = 'resnet50'
# Annotation файлууд
annotations_path = './an.csv'
classes_path = './class.csv'
validations_path = './validation.csv'
evaluation = True
# Хэрэв сургалтаа үргэлжлүүлэх бол snapshot-ын утганд сүүлийн сургалтын файлыг зааж өгнө
snapshot = None
# Snapshot хадгалах эсэх, хадгалах хавтас
snapshots = True
snapshot_path = './snapshots'

tensorboard_dir = './logs'

weights = None
imagenet_weights = True
batch_size = 8
image_min_side = 375
image_max_side = 500


backbone_model = models.backbone('resnet50')
freeze_backbone = False
common_args = {
    'batch_size'       : batch_size,
    'image_min_side'   : image_min_side,
    'image_max_side'   : image_max_side,
    'preprocess_image' : backbone_model.preprocess_image,
}

train_generator = CSVGenerator(annotations_path, classes_path, **common_args)
if validations_path != '':
    validation_generator = CSVGenerator(validations_path, classes_path, **common_args)
else:
    validation_generator = None

# Model-оо үүсгэх эсвэл өмнө нь үүссэн model-г унших
if snapshot is not None:
    print('Loading model, this may take a second...')
    model = models.load_model(snapshot, backbone_name=backbone)
    training_model = model
    prediction_model = retinanet_bbox(model=model)
else:
    weights = weights
    # default to imagenet if nothing else is specified
    if weights is None and imagenet_weights:
        weights = backbone_model.download_imagenet()

    print('Creating model, this may take a second...')
    modifier = freeze_model if freeze_backbone else None
    
    model = backbone_model.retinanet(train_generator.num_classes(), modifier=modifier)
    if weights is not None:
        model.load_weights(weights, by_name=True, skip_mismatch=True)
    training_model = model

    # make prediction model
    prediction_model = retinanet_bbox(model=model)

    # compile model
    training_model.compile(
        loss={
            'regression'    : losses.smooth_l1(),
            'classification': losses.focal()
        },
        optimizer=keras.optimizers.adam(lr=1e-5, clipnorm=0.001)
    )

# model-н тухай хураангуй
print(model.summary())

# this lets the generator compute backbone layer shapes using the actual backbone model
if 'vgg' in backbone or 'densenet' in backbone:
    train_generator.compute_shapes = make_shapes_callback(model)
    if validation_generator:
        validation_generator.compute_shapes = train_generator.compute_shapes


callbacks = []

tensorboard_callback = None

if tensorboard_dir:
    tensorboard_callback = keras.callbacks.TensorBoard(
        log_dir                = tensorboard_dir,
        histogram_freq         = 0,
        batch_size             = batch_size,
        write_graph            = True,
        write_grads            = False,
        write_images           = False,
        embeddings_freq        = 0,
        embeddings_layer_names = None,
        embeddings_metadata    = None
    )
    callbacks.append(tensorboard_callback)

if evaluation and validation_generator:
    evaluation = Evaluate(validation_generator, tensorboard=tensorboard_callback)
    evaluation = RedirectModel(evaluation, prediction_model)
    #callbacks.append(evaluation)

# model хадгалах
if snapshots:
    # хадгалах хавтсыг үүсгэх
    try:
        os.makedirs(snapshot_path)
    except OSError:
        if not os.path.isdir(snapshot_path):
            raise
    checkpoint = keras.callbacks.ModelCheckpoint(
        os.path.join(
            snapshot_path,
            '{backbone}_{dataset_type}_{{epoch:02d}}.h5'.format(backbone=backbone, dataset_type='csv')
        ),
        verbose=1,
        # save_best_only=True,
        # monitor="mAP",
        # mode='max'
    )
    checkpoint = RedirectModel(checkpoint, model)
    callbacks.append(checkpoint)

callbacks.append(keras.callbacks.ReduceLROnPlateau(
    monitor  = 'loss',
    factor   = 0.1,
    patience = 2,
    verbose  = 1,
    mode     = 'auto',
    epsilon  = 0.0001,
    cooldown = 0,
    min_lr   = 0
))


epochs = 5
steps = 100


training_model.fit_generator(
    generator=train_generator,
    steps_per_epoch=steps,
    epochs=epochs,
    verbose=1,
    callbacks=callbacks,
)
