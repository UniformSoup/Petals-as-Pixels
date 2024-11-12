%% Change all lables != one (flower) to zero (background), and remove images with no label.

% Get all the original filenames.
labelDir = "data_for_moodle/labels_256";
newLabelDir = "data_for_moodle_preprocessed/labels_256";
imageDir = "data_for_moodle/images_256";
newImageDir = "data_for_moodle_preprocessed/images_256";
fileList = dir(fullfile(labelDir, '*.png'));

% Iterate over each file.
for i = 1 : numel(fileList)
    % Read the label and colourmap in.
    fileName = fileList(i).name;
    [labelImage, labelColorMap] = imread(fullfile(labelDir, fileName));
    
    % Modify labels
    labelImage(labelImage ~= 1) = 0;
    
    % Save the modified image (w/ colormap).
    imwrite(labelImage, labelColorMap, fullfile(newLabelDir, fileName));

    % Copy image with this filename.
    imageFileName = strrep(fileName, 'png', 'jpg'); 
    copyfile(fullfile(imageDir, imageFileName), fullfile(newImageDir, imageFileName));
end

labels = pixelLabelDatastore(newLabelDir, ["Flower", "Background"], [1, 0]);
labelCounts = countEachLabel(labels);
histogram('Categories', labelCounts.Name, 'BinCounts', labelCounts.PixelCount, 'Normalization', 'probability')
xlabel('Label')
ylabel('Normalised Pixel Count')
saveas(gcf, 'Plots/distribution.eps', 'epsc')
