using System;

#nullable enable
namespace Neural_Network
{
	class CostOptions
	{
		public QuadraticCost Quadratic = new QuadraticCost();
		public CrossEntropyCost CrossEntropy = new CrossEntropyCost();

		public class QuadraticCost : ICost
		{
			public Matrix Function(Matrix output, Matrix y)
			{
				Matrix error = Matrix.Subtract(output, y);
				return Matrix.Scale(Matrix.HadamardProduct(error, error), 0.5d);
			}

			public Matrix Delta(Matrix output, Matrix y, Matrix? z = null, Func<double, double>? dActivator = null)
			{
				return Matrix.HadamardProduct(Matrix.Subtract(output, y), Matrix.Evaluate(z, dActivator));
			}
		}

		public class CrossEntropyCost : ICost
		{
			private double OneMinus(double x)
			{
				return 1d - x;
			}
			public Matrix Function(Matrix output, Matrix y)
			{
				return Matrix.Scale(Matrix.Add(Matrix.HadamardProduct(y, Matrix.Evaluate(output, Math.Log)), Matrix.HadamardProduct(Matrix.Evaluate(y, OneMinus), Matrix.Evaluate(Matrix.Evaluate(output, OneMinus), Math.Log))), -1d);
			}

			public Matrix Delta(Matrix output, Matrix y, Matrix? z = null, Func<double, double>? dActivator = null)
			{
				return Matrix.Subtract(output, y);
			}
		}
	}
}
#nullable disable
