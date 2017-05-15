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



validation_generator = test_datagen.flow_from_directory('/home/ec2-user/test/1/',
                                                        class_mode=None,
                                                        target_size=loaded_model.input_shape[1:3],
                                                        color_mode="grayscale",
                                                        shuffle=False,
                                                        batch_size=10000)

predictions = loaded_model.predict_classes(validation_generator.next())
import re
z = zip(validation_generator.filenames, predictions)
zz = map(lambda x: (int(re.findall(r'\d+', x[0])[0]), 2-x[1]), z)
zzz = sorted(zz, key=lambda x:x[0])
for i in zzz:
    print i[1][0]

# a = {i.split(".")[0].split("/")[1]: j[0] for i, j in zip(validation_generator.filenames, predictions)}
# b = {int(re.find(r'\d+',i)[0]): j for i, j in a.items()}
# c = {i: 2 if j < 0.5 else 1 for i, j in b.items()}
# result = [i[1] for i in sorted(c.items(), key=lambda x: x[0])]
# print("\n".join(map(str, result)))
