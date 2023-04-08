namespace Neural_Network
{
	interface IActivation
	{
		public double Function(double x);
		public double Derivative(double x);
	}
}
