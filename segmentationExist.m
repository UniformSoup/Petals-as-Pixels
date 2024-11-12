%% Load dataset into memory.
imageDir = fullfile("data_for_moodle_preprocessed/images_256");
labelDir = fullfile("data_for_moodle_preprocessed/labels_256");

% Ground truth labels.
% All classes that aren't flower are set to zero in the preprocessing.
classes = ["Flower","Background"];
labelIDs = [1 0];

% Create Datastores.
images = dir(fullfile(imageDir, '*.jpg'));
images = fullfile(imageDir, {images.name});
labels = dir(fullfile(labelDir, '*.png'));
labels = fullfile(labelDir, {labels.name});

% Split. (0.9, 0.05, 0.05)
[training, validation] = split(randperm(length(images)), 0.9);
[validation, testing] = split(validation, 0.5);

% Create datasets.
trainingImages = imageDatastore(images(training));
trainingLabels = pixelLabelDatastore(labels(training), classes, labelIDs);
training = combine(trainingImages, trainingLabels);

validationImages = imageDatastore(images(validation));
validationLabels = pixelLabelDatastore(labels(validation), classes, labelIDs);
validation = combine(validationImages, validationLabels);

testingImages = imageDatastore(images(testing));
testingLabels = pixelLabelDatastore(labels(testing), classes, labelIDs);
testing = combine(testingImages, testingLabels);

counts = countEachLabel(trainingLabels);
classWeights = sum(counts.PixelCount) ./ counts.PixelCount;

%% Create Model.
net = deeplabv3plus([256 256], length(classes), "resnet18");

%% Train Model.

options = trainingOptions('sgdm', ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropPeriod', 3, ...
    'LearnRateDropFactor', 0.2, ...
    'InitialLearnRate', 0.001, ...
    'GradientThreshold', 0.5, ...
    'MaxEpochs', 30, ...
    'Plots', "training-progress", ...
    'Metrics', ["accuracy", "fscore"], ...
    'ValidationData', validation, ... 
    'ValidationPatience', 5, ...
    'ValidationFrequency', 10, ...
    'MiniBatchSize', 16, ...
    'Shuffle', 'never', ...
    'L2Regularization', 0.01, ...
    'ExecutionEnvironment', 'parallel');

[net, info] = trainnet(training, net, @(pred, target) crossentropy(pred, target, classWeights, WeightsFormat="C"), options);
%save the network in case we want to use it again
stamp = datestr(now,'HH.MM.SS.FFF')
save(['segmentexistnet.', stamp, '.mat'], 'net')
save(['segmentexistnet.info.', stamp, '.mat'], 'info')
%% Results.
testingResults = semanticseg(testingImages,net,Classes=classes,WriteLocation=tempdir);
results = evaluateSemanticSegmentation(testingResults,testingLabels);
save(['segmentexistnet.results.', stamp, '.mat'], 'results')

% Read an Example
test_image = imread("data_for_moodle_preprocessed/images_256/image_0001.jpg");
% Plot the Image
imshow(test_image)
% Evaluate on network
prediction = semanticseg(test_image, net);
% Overlay on Image.
imshow(labeloverlay(test_image, prediction))

%% Function definitions.

function [trainIndices, testIndices] = split(indices, ratio)    
    % Calculate the number of indices for the training and testing sets
    split = floor(ratio * length(indices));
    
    % Assign indices to the training and testing sets
    trainIndices = indices(1 : split);
    testIndices = indices(split + 1 : end);
end