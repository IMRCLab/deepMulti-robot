DATASET_FOLDER = '/home/akmaral/tubCloud/Shared/cvmrs/training_dataset/synth/single/1k/raw-images/'
# DATASET_FOLDER = 'aideck-dataset/imageStorage'
INPUT_CHANNEL = 1 # RGB: 3, Grey: 1
LOCA_STRIDE     = 8
LOCA_CLASSES    = {0: "crazyflie"}
TRAIN_ANNOT_PATH    = "{}../train.txt".format(DATASET_FOLDER)
TEST_PATH    = "{}../test.txt".format(DATASET_FOLDER)
TRAIN_BATCH_SIZE    = 8
TRAIN_INPUT_SIZE    = [320, 320]
TRAIN_LR_INIT       = 1e-3
TRAIN_LR_END        = 1e-6
TRAIN_WARMUP_EPOCHS = 2
TRAIN_EPOCHS        = 25
OUTPUT_FILE = 'synth-1k'
WEIGHT_PATH = '/home/akmaral/tubCloud/Shared/cvmrs/trained-models/locanet/'