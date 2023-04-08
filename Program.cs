using System;
using System.Diagnostics;

namespace Neural_Network
{
	class Program
	{
		static void Main(string[] args)
		{
			Utils.Init(DateTime.Now.Second);
			Console.CursorVisible = false;
			Console.Title = "Neural Network";
			Console.SetWindowSize(50, 50);
			Console.SetWindowPosition(0, 0);

			Matrix LOW_LOW   = new Matrix(0d, 0d);
			Matrix LOW_HIGH  = new Matrix(0d, 1d);
			Matrix HIGH_LOW  = new Matrix(1d, 0d);
			Matrix HIGH_HIGH = new Matrix(1d, 1d);
			Matrix HIGH      = new Matrix(1d);
			Matrix LOW       = new Matrix(0d);

			Stopwatch sw;

			(Matrix input, Matrix y)[] trainingData = new (Matrix, Matrix)[] {
				(HIGH_HIGH, LOW),
				(HIGH_LOW, HIGH),
				(LOW_HIGH, HIGH),
				(LOW_LOW, LOW)
			};

			NeuralNetwork nn = new NeuralNetwork(NeuralNetwork.Costs.CrossEntropy, 0.005d, (-1d, 1d), (-1d, 1d), 2, (2, NeuralNetwork.Activations.Sigmoid), (2, NeuralNetwork.Activations.Sigmoid), (1, NeuralNetwork.Activations.Sigmoid));
			Console.WriteLine("Press enter to begin.");
			Console.ReadLine();
			Console.Clear();
			for (int i = 0; i < 1000000; i++)
			{
				//sw = Stopwatch.StartNew();
				nn.TrainMiniBatch(trainingData, 4);
				//sw.Stop();
				//Console.WriteLine($"Epoch: {i}, potential {1d / (sw.ElapsedTicks / 10000000d):F0} epochs/s (without console output)");
				if (i % 10000 == 0)
				{
					Console.SetCursorPosition(0, 0);
					nn.PrintWeights();
					nn.PrintBiases();
					nn.FeedForward(HIGH_HIGH).Print("1, 1: ");
					nn.FeedForward(HIGH_LOW).Print("1, 0: ");
					nn.FeedForward(LOW_HIGH).Print("0, 1: ");
					nn.FeedForward(LOW_LOW).Print("0, 0: ");
					Console.WriteLine($"Epoch {i}.");
				}
			}

			Console.ReadLine();
		}
	}
}
