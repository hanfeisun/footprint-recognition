import argparse

from keras.preprocessing.image import ImageDataGenerator

parser = argparse.ArgumentParser(description='Process some integers.')

parser.add_argument('-w', '--weight', type=str, help='weight h5 file')
parser.add_argument('-v', '--validation', type=str, help='validation/test file')

args = parser.parse_args()

# json_file = open(args.model, 'r')
# loaded_model_json = json_file.read()
# json_file.close()


from keras.models import model_from_json, load_model

loaded_model = load_model(args.weight)
#
# # loaded_model.load_weights(args.weight)
test_datagen = ImageDataGenerator(rescale=1. / 255)

validation_generator = test_datagen.flow_from_directory('../3/',
                                                        class_mode=None,
                                                        target_size=loaded_model.input_shape[1:3],
                                                        color_mode="grayscale",
                                                        shuffle=False,
                                                        batch_size=10000)

# predictions = loaded_model.predict_generator(validation_generator, validation_generator.n, verbose=1)
raise
predictions = loaded_model.predict_classes(validation_generator.next())

print(predictions)
import re
z = zip(validation_generator.filenames, predictions)
zz = map(lambda x: (int(re.findall(r'\d+', x[0])[0]), x[1]+1), z)
zzz = sorted(zz, key=lambda x:x[0])
for i in zzz:
    print i[1]

