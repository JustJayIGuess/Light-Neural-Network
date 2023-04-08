using System;

namespace Neural_Network
{
	static class Utils
	{
		private static Random rand;
		private static readonly object syncLock = new object();

		public static void Init(int seed = 0)
		{
			if (rand == null)
			{
				rand = new Random(seed);
			}
		}
		public static int Random()
		{
			Init();
			lock (syncLock)
			{
				return rand.Next();
			}
		}
		public static int Random(int max)
		{
			Init();
			lock (syncLock)
			{
				if (max == 0)
				{
					return 0;
				}
				return rand.Next(max);
			}
		}
		public static int Random(int min, int max)
		{
			Init();
			lock (syncLock)
			{
				return rand.Next(min, max);
			}
		}
		public static double Random(double max)
		{
			Init();
			lock (syncLock)
			{
				return rand.NextDouble() * max;
			}
		}
		private static double Map(double value, double fromLow, double fromHigh, double toLow, double toHigh)
		{
			return (value - fromLow) * (toHigh - toLow) / (fromHigh - fromLow) + toLow;
		}
		public static double Random(double min, double max)
		{
			Init();
			lock (syncLock)
			{
				return Map(rand.NextDouble(), 0d, 1d, min, max);
			}
		}
	}
}
