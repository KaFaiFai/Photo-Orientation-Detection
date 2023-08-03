# Photo-Orientation-Detection
Classify input photos into four classes: 0◦, 90◦, 180◦, or 270◦, and rotate them accordingly

## Setup
1. go to https://www.cityscapes-dataset.com/downloads/ and download the `gtFine_trainvaltest.zip` and `leftImg8bit_trainvaltest.zip`
2. unzip them to `./training_data`

## Todo
1. ~~Add accuracy metrics~~ :white_check_mark:
1. ~~Add save model function~~ :white_check_mark:
1. ~~reduce gpu memory usage~~ :white_check_mark:
1. Add max_size for image data
1. Implement EfficientNet https://paperswithcode.com/paper/efficientnet-rethinking-model-scaling-for
1. Implement MobileVit https://paperswithcode.com/paper/mobilevit-light-weight-general-purpose-and
1. Support training with pretrained model for MobileNetV2
1. Add evaluate model script
1. Add config file
1. Objectify dataset looper