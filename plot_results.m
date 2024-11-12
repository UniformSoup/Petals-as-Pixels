% Load and plot saved results for bespoke network
load('segmentownnet.mat');
load('segmentownnet.info.mat');
load('segmentownnet.results.mat');
plot_all_results(info.TrainingHistory, info.ValidationHistory, results, net, 'own')

% Load and plot saved results for existing network
load('segmentexistnet.mat');
load('segmentexistnet.info.mat');
load('segmentexistnet.results.mat');
plot_all_results(info.TrainingHistory, info.ValidationHistory, results, net, 'exist')

function plot_all_results(trainHist, valHist, results, net, type)
    % Plot the loss
    figure;
    plot(trainHist.Iteration, trainHist.Loss, 'b');
    hold on;
    plot(valHist.Iteration, valHist.Loss, 'r');
    hold off;
    xlabel('Iterations');
    ylabel('Loss');
    legend('Training', 'Validation', 'Location', 'northeast');
    saveas(gcf, ['Plots/', type, '_loss.eps'], 'epsc');
    
    % Plot F1 score
    figure;
    plot(trainHist.Iteration, trainHist.FScore, 'b');
    hold on;
    plot(valHist.Iteration, valHist.FScore, 'r');
    hold off;
    xlabel('Iterations');
    ylabel('F1 Score');
    legend('Training', 'Validation', 'Location', 'southeast');
    saveas(gcf, ['Plots/', type, '_metrics.eps'], 'epsc');
    
    % Plot the confusion matrix
    figure;
    confusionchart(table2array(results.ConfusionMatrix), {'Flower', 'Background'}, 'Normalization', 'row-normalized');
    xlabel('Predicted');
    ylabel('True');
    saveas(gcf, ['Plots/', type, '_confusion_matrix.eps'], 'epsc');
    
    % Plot test image
    figure;
    test_image = imread("data_for_moodle_preprocessed/images_256/image_0001.jpg");
    imshow(test_image)
    
    % Evaluate with segnet and overlay
    prediction = semanticseg(test_image, net);
    imshow(labeloverlay(test_image, prediction))
    imwrite(labeloverlay(test_image, prediction), ['Plots/', type,  '_example_result.png']);

    figure;
    plot(net)
    saveas(gcf, ['Plots/', type, '_architecture.eps'], 'epsc');
end
