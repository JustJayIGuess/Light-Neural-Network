using System;

namespace Neural_Network
{
	class ActivationOptions
	{
		public SigmoidActivation Sigmoid = new SigmoidActivation();
		public NoActivation None = new NoActivation();
		public ReLUActivation ReLU = new ReLUActivation();
		public class SigmoidActivation : IActivation
		{
			public double Function(double x)
			{
				return 1d / (1d + Math.Exp(-x));
			}
			public double Derivative(double x)
			{
				double val = Function(x);
				return val * (1d - val);
			}
		}

		public class ReLUActivation : IActivation
		{
			public double Function(double x)
			{
				return x < 0d ? 0d : x;
			}
			public double Derivative(double x)
			{
				return x < 0d ? 0d : 1d;
			}
		}

		public class NoActivation : IActivation
		{
			public double Function(double x)
			{
				return x;
			}
			public double Derivative(double x)
			{
				return 1d;
			}
		}

	}
}
