using System;

namespace Neural_Network
{
	class NeuralNetwork
	{
		private Matrix[] weightMatrices;
		private Matrix[] biasMatrices;
		private Matrix[] activationMatrices;
		private Matrix[] preActivationMatrices;

		private Matrix[] weightMatricesTransposedCache;
		private Matrix[] errorMatricesCache;
		private Matrix[] backpropCache;
		private Matrix[] delBiasMatricesCache;
		private Matrix[] delWeightMatricesCache;

		private readonly Func<double, double>[] activators;
		private readonly Func<double, double>[] dActivators;

#nullable enable
		public delegate Matrix DeltaDelegate(Matrix output, Matrix y, Matrix? z = null, Func<double, double>? dActivator = null);
		private readonly Func<Matrix, Matrix, Matrix> cost;
		private DeltaDelegate delta;
#nullable disable

		public double LearningRate { get; set; }
		public static ActivationOptions Activations = new ActivationOptions();
		public static CostOptions Costs = new CostOptions();

		public NeuralNetwork(ICost costFunc, double learningRate, (double, double) weightRange, (double, double) biasRange, int inputSize, params (int layerSize, IActivation layerActivation)[] layerInfo)
		{
			if (layerInfo.Length < 2)
			{
				throw new Exception("ERROR: Not enough layers to create neural network!");
			}
			LearningRate = learningRate;
			weightMatrices = new Matrix[layerInfo.Length];
			biasMatrices = new Matrix[layerInfo.Length];
			activationMatrices = new Matrix[layerInfo.Length];
			preActivationMatrices = new Matrix[layerInfo.Length];

			weightMatricesTransposedCache = new Matrix[layerInfo.Length];
			errorMatricesCache = new Matrix[layerInfo.Length];
			backpropCache = new Matrix[layerInfo.Length];
			delBiasMatricesCache = new Matrix[layerInfo.Length];
			delWeightMatricesCache = new Matrix[layerInfo.Length];

			activators = new Func<double, double>[layerInfo.Length];
			dActivators = new Func<double, double>[layerInfo.Length];

			cost = costFunc.Function;
			delta = costFunc.Delta;

			for (int i = 0; i < layerInfo.Length; i++)
			{
				activators[i] = layerInfo[i].layerActivation.Function;
				dActivators[i] = layerInfo[i].layerActivation.Derivative;

				weightMatrices[i] = new Matrix(layerInfo[i].layerSize, i == 0 ? inputSize : layerInfo[i - 1].layerSize);
				weightMatrices[i].Randomize(weightRange.Item1, weightRange.Item2);
				biasMatrices[i] = new Matrix(layerInfo[i].layerSize);
				biasMatrices[i].Randomize(biasRange.Item1, biasRange.Item2);
				preActivationMatrices[i] = new Matrix(layerInfo[i].layerSize);
				preActivationMatrices[i].InitializeWithValues(0d);
				activationMatrices[i] = new Matrix(layerInfo[i].layerSize);

				weightMatricesTransposedCache[i] = new Matrix(i == 0 ? inputSize : layerInfo[i - 1].layerSize, layerInfo[i].layerSize);
				errorMatricesCache[i] = new Matrix(layerInfo[i].layerSize);
				backpropCache[i] = new Matrix(layerInfo[i].layerSize);
				delBiasMatricesCache[i] = new Matrix(layerInfo[i].layerSize);
				delWeightMatricesCache[i] = new Matrix(layerInfo[i].layerSize, i == 0 ? inputSize : layerInfo[i - 1].layerSize);
			}
		}

		public void PrintWeights()
		{
			Console.WriteLine("Weights:");
			for (int i = 0; i < weightMatrices.Length; i++)
			{
				Console.WriteLine($"  Layer {i + 1}:");
				weightMatrices[i].Print("    ");
			}
		}

		public void PrintBiases()
		{
			Console.WriteLine("Biases:");
			for (int i = 0; i < biasMatrices.Length; i++)
			{
				Console.WriteLine($"  Layer {i + 1}:");
				biasMatrices[i].Print("    ");
			}
		}

		public void PrintNeurons()
		{
			Console.WriteLine("Neurons:");
			for (int i = 0; i < activationMatrices.Length; i++)
			{
				Console.WriteLine($"  Layer {i + 1}:");
				activationMatrices[i].Print("    ");
			}
		}

		// TODO: Stop initialising new matrices every time.
		public Matrix FeedForward(Matrix input)
		{
			preActivationMatrices[0].InitializeWithValues(0d);

			preActivationMatrices[0].Add(Matrix.Multiply(weightMatrices[0], input));
			preActivationMatrices[0].Add(biasMatrices[0]);
			activationMatrices[0].SetEvaluated(preActivationMatrices[0], activators[0]);
			for (int i = 1; i < activationMatrices.Length; i++)
			{
				preActivationMatrices[i].InitializeWithValues(0d);

				preActivationMatrices[i].Add(Matrix.Multiply(weightMatrices[i], activationMatrices[i - 1]));    // Note: allocation (required i think)
				preActivationMatrices[i].Add(biasMatrices[i]);
				activationMatrices[i].SetEvaluated(preActivationMatrices[i], activators[i]);
			}
			return activationMatrices[^1];
		}

		/// <summary>
		/// Warning: this returns a reference to cached objects - please use this data before calling other functions that may edit these obects,
		/// </summary>
		/// <param name="input"></param>
		/// <param name="y"></param>
		/// <returns></returns>
		private (Matrix[] delBs, Matrix[] delWs) Backprop(Matrix input, Matrix y)
		{
			// Feedforward
			Matrix output = FeedForward(input);

			// Last-layer error
			errorMatricesCache[^1] = delta(output, y, preActivationMatrices[^1], dActivators[^1]);

			// Backprop error
			for (int l = errorMatricesCache.Length - 2; l >= 0; l--)
			{
				weightMatricesTransposedCache[l + 1].SetTransposedCopy(weightMatrices[l + 1]);

				errorMatricesCache[l].SetEvaluated(preActivationMatrices[l], dActivators[l]);
				errorMatricesCache[l].HadamardProduct(Matrix.Multiply(weightMatricesTransposedCache[l + 1], errorMatricesCache[l + 1]));
			}

			// Calculate gradients

			//     Biases
			for (int l = 0; l < delBiasMatricesCache.Length; l++)
			{
				delBiasMatricesCache[l].SetCopy(errorMatricesCache[l]);
			}

			//     Weights
			for (int l = 0; l < delWeightMatricesCache.Length; l++)
			{
				//delWeightMatricesCache[l] = new Matrix(weightMatrices[l].Rows, weightMatrices[l].Cols);
				for (int row = 0; row < delWeightMatricesCache[l].Rows; row++)
				{
					for (int col = 0; col < delWeightMatricesCache[l].Cols; col++)
					{
						if (l == 0)
						{
							delWeightMatricesCache[l][row, col] = input[col] * errorMatricesCache[l][row];
						}
						else
						{
							delWeightMatricesCache[l][row, col] = activationMatrices[l - 1][col] * errorMatricesCache[l][row];
						}
					}
				}
			}

			return (delBiasMatricesCache, delWeightMatricesCache);
		}

		public void TrainStochastic((Matrix input, Matrix y)[] trainingData)
		{
			(Matrix input, Matrix y) trainingPoint = trainingData[Utils.Random(trainingData.Length)];
			(Matrix[] delBs, Matrix[] delWs) gradient = Backprop(trainingPoint.input, trainingPoint.y);
			for (int l = 0; l < biasMatrices.Length; l++)
			{
				biasMatrices[l].AddScaled(gradient.delBs[l], -LearningRate);
				weightMatrices[l].AddScaled(gradient.delWs[l], -LearningRate);
			}
		}

		public void TrainSubset((Matrix input, Matrix y)[] trainingData)
		{
			(Matrix[] delBs, Matrix[] delWs) gradient;
			(Matrix[] delBs, Matrix[] delWs) gradientTemp;
			gradient = Backprop(trainingData[0].input, trainingData[0].y);
			for (int i = 1; i < trainingData.Length; i++)
			{
				gradientTemp = Backprop(trainingData[i].input, trainingData[i].y);
				for (int l = 0; l < gradient.delBs.Length; l++)
				{
					gradient.delBs[l].Add(gradientTemp.delBs[l]);
					gradient.delWs[l].Add(gradientTemp.delWs[l]);
				}
			}
			for (int l = 0; l < gradient.delBs.Length; l++)
			{
				biasMatrices[l].AddScaled(gradient.delBs[l], -LearningRate / trainingData.Length);
				weightMatrices[l].AddScaled(gradient.delWs[l], -LearningRate / trainingData.Length);
			}
		}

		public void TrainMiniBatch((Matrix input, Matrix y)[] trainingData, int batchSize)
		{
			(Matrix input, Matrix y)[] miniBatch = new (Matrix, Matrix)[batchSize];
			for (int i = 0; i < miniBatch.Length; i++)
			{
				miniBatch[i] = trainingData[Utils.Random(trainingData.Length)];
			}
			TrainSubset(miniBatch);
		}
	}
}
