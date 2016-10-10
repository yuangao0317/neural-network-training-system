function p = runPrediction(file_name, Theta1, Theta2)

x = dlmread (file_name, ",");
p = predict(Theta1, Theta2, x);

fprintf('\nThe number is: %d\n', p);

end
