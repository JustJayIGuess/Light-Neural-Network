using System;
#nullable enable

namespace Neural_Network
{
	interface ICost
	{
		public Matrix Function(Matrix output, Matrix y);
		public Matrix Delta(Matrix output, Matrix y, Matrix? z = null, Func<double, double>? dActivator = null);
	}
}
#nullable disable
