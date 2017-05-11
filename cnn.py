from keras.models import load_model, Model
from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Flatten
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD
from os.path import dirname
from os import listdir
from sys import argv

TRAIN_DIR = './data/train/'
VAL_DIR = './data/val/'
img_size = 96
mini_batch_sz = 16
nb_train_samples = sum([len(listdir(TRAIN_DIR + CLASS_NAME)) for CLASS_NAME in listdir(TRAIN_DIR)])
nb_val_samples = sum([len(listdir(VAL_DIR + CLASS_NAME)) for CLASS_NAME in listdir(VAL_DIR)])

def visualizer(model):
	from keras.utils import plot_model
	plot_model(model, to_file='vis.png', show_shapes=True)

def init_model(preload, declare=False):
	if not declare and preload:
		return load_model(preload)

	img_input = Input(shape=(img_size, img_size, 3))
	x = Conv2D(16, (3, 3), padding='same', activation='relu') (img_input)
	x = BatchNormalization(axis=3) (x)
	x = MaxPooling2D((3, 3), strides=(3, 3)) (x)
	x = Conv2D(32, (3, 3), padding='same', activation='relu') (x)
	x = BatchNormalization(axis=3) (x)
	x = MaxPooling2D((2, 2), strides=(2, 2)) (x)
	x = Flatten() (x)
	x = Dense(16, activation='relu') (x)
	x = BatchNormalization() (x)
	prediction = Dense(12, activation='softmax') (x)

	model = Model(img_input, prediction)

	if preload:
		model.load_weights(preload)
	return model

def DataGen():
	train_gen = ImageDataGenerator(zoom_range=0.01, shear_range=0.01, rotation_range=5., channel_shift_range=1.)
	val_gen = ImageDataGenerator()

	train_gen = train_gen.flow_from_directory(
		TRAIN_DIR,
		target_size=(img_size, img_size),
		batch_size=mini_batch_sz,
		class_mode='categorical'
		)

	val_gen = val_gen.flow_from_directory(
		VAL_DIR,
		target_size=(img_size, img_size),
		batch_size=mini_batch_sz,
		class_mode='categorical',
		shuffle=False
		)

	return train_gen, val_gen

def runner(model, epochs=100):
	initial_LR = 1e-3

	train_gen, val_gen = DataGen()

	model.compile(optimizer=SGD(initial_LR, momentum=0.99, nesterov=True), loss='categorical_crossentropy', metrics=['acc'])

	val_checkpoint = ModelCheckpoint('bestval.h5', 'val_loss', 1, True)
	cur_checkpoint = ModelCheckpoint('current.h5')

	print 'Model compiled'

	model.fit_generator(generator=train_gen, steps_per_epoch=nb_train_samples // mini_batch_sz, epochs=epochs,
						verbose=1, validation_data=val_gen, validation_steps=nb_val_samples // mini_batch_sz,
						callbacks=[val_checkpoint, cur_checkpoint])

def main(args):
	mode, preload = args
	if preload == 'none': preload = None
	if mode == 'vis':
		return visualizer(init_model(preload))
	if mode == 'train':
		return runner(init_model(preload))

if __name__ == '__main__':
	main(argv[1:])