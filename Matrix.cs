using Microsoft.VisualBasic.CompilerServices;
using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace Neural_Network
{
	class Matrix
	{
		private double[,] data;

		public int Rows { get; private set; }
		public int Cols{ get; private set; }

		public Matrix(int rows, int cols)
		{
			data = new double[rows, cols];
			Rows = rows;
			Cols = cols;
		}

		public Matrix(int rows)
		{
			data = new double[rows, 1];
			Rows = rows;
			Cols = 1;
		}

		public Matrix(double[,] _data)
		{
			data = _data;
			Rows = data.GetLength(0);
			Cols = data.GetLength(1);
		}

		public Matrix(params double[] _data)
		{
			Rows = _data.Length;
			Cols = 1;
			data = new double[Rows, 1];
			for (int i = 0; i < _data.Length; i++)
			{
				data[i, 0] = _data[i];
			}
		}


		public double this[int row, int col]
		{
			get => data[row, col];
			set => data[row, col] = value;
		}

		public double this[int row]
		{
			get => data[row, 0];
			set => data[row, 0] = value;
		}

		public void Print(string prepend = "")
		{
			for (int i = 0; i < Rows; i++)
			{
				Console.Write(prepend);
				for (int j = 0; j < Cols; j++)
				{
					double datum = data[i, j];
					if (!double.IsNegative(datum))
					{
						Console.Write(" ");
					}
					Console.Write($"{datum:F3}, ");
				}
				Console.WriteLine();
			}
			Console.WriteLine();
		}

		public void Randomize(double min, double max)
		{
			for (int i = 0; i < Rows; i++)
			{
				for (int j = 0; j < Cols; j++)
				{
					data[i, j] = Utils.Random(min, max);
				}
			}
		}

		public Matrix Subtract(Matrix m)
		{
			if (m.Rows != Rows || m.Cols != Cols)
			{
				throw new Exception("ERROR: Tried to subtract matrixes of different size!");
			}
			for (int i = 0; i < Rows; i++)
			{
				for (int j = 0; j < Cols; j++)
				{
					data[i, j] -= m[i, j];
				}
			}
			return this;
		}

		public Matrix Subtract(double n)
		{
			for (int i = 0; i < Rows; i++)
			{
				for (int j = 0; j < Cols; j++)
				{
					data[i, j] -= n;
				}
			}
			return this;
		}

		public Matrix Add(Matrix m)
		{
			if (m.Rows != Rows || m.Cols != Cols)
			{
				throw new Exception("ERROR: Tried to add matrixes of different size!");
			}
			for (int i = 0; i < Rows; i++)
			{
				for (int j = 0; j < Cols; j++)
				{
					data[i, j] += m[i, j];
				}
			}
			return this;
		}

		public Matrix AddScaled(Matrix m, double n)
		{
			if (m.Rows != Rows || m.Cols != Cols)
			{
				throw new Exception("ERROR: Tried to add matrixes of different size!");
			}
			for (int i = 0; i < Rows; i++)
			{
				for (int j = 0; j < Cols; j++)
				{
					data[i, j] += m[i, j] * n;
				}
			}
			return this;
		}

		public Matrix Add(double n)
		{
			for (int i = 0; i < Rows; i++)
			{
				for (int j = 0; j < Cols; j++)
				{
					data[i, j] += n;
				}
			}
			return this;
		}

		public Matrix Scale(double n)
		{
			for (int i = 0; i < Rows; i++)
			{
				for (int j = 0; j < Cols; j++)
				{
					data[i, j] *= n;
				}
			}
			return this;
		}

		public void InitializeWithValues(double val)
		{
			for (int i = 0; i < Rows; i++)
			{
				for (int j = 0; j < Cols; j++)
				{
					data[i, j] = val;
				}
			}
		}

		public void SetCopy(Matrix m)
		{
			for (int i = 0; i < Rows; i++)
			{
				for (int j = 0; j < Cols; j++)
				{
					data[i, j] = m[i, j];
				}
			}
		}

		public void SetTransposedCopy(Matrix m)
		{
			for (int i = 0; i < Rows; i++)
			{
				for (int j = 0; j < Cols; j++)
				{
					data[i, j] = m[j, i];
				}
			}
		}

		public Matrix Evaluate(Func<double, double> mapper)
		{
			for (int i = 0; i < Rows; i++)
			{
				for (int j = 0; j < Cols; j++)
				{
					data[i, j] = mapper(data[i, j]);
				}
			}
			return this;
		}

		public void Multiply(Matrix m)
		{
			if (m.Cols != Rows)
			{
				throw new Exception("Tried to multiply unmultiplicable matrices in Matrix.Multiply!");
			}
			for (int i = 0; i < m.Rows; i++)
			{
				for (int j = 0; j < Cols; j++)
				{
					for (int k = 0; k < m.Cols; k++)
					{
						data[i, j] += m[i, k] * data[k, j];
					}
				}
			}
		}

		public void SetEvaluated(Matrix m, Func<double, double> mapper)
		{
			for (int i = 0; i < m.Rows; i++)
			{
				for (int j = 0; j < m.Cols; j++)
				{
					data[i, j] = mapper(m.data[i, j]);
				}
			}
		}

		public void HadamardProduct(Matrix b)
		{
			for (int i = 0; i < Rows; i++)
			{
				for (int j = 0; j < Cols; j++)
				{
					data[i, j] *= b[i, j];
				}
			}

		}

		public static Matrix HadamardProduct(Matrix a, Matrix b)
		{
			if (a.Rows != b.Rows || a.Cols != b.Cols)
			{
				throw new Exception("Tried to multiply unmultiplicable matrices in Matrix.HadamardProduct!");
			}

			Matrix res = new Matrix(a.Rows, a.Cols);

			for (int i = 0; i < res.Rows; i++)
			{
				for (int j = 0; j < res.Cols; j++)
				{
					res[i, j] = a[i, j] * b[i, j];
				}
			}

			return res;
		}

		public static Matrix RandomOfSize((int rows, int cols) size, (double min, double max) range)
		{
			Matrix res = new Matrix(size.rows, size.cols);
			res.Randomize(range.min, range.max);
			return res;
		}

		public static Matrix Multiply(Matrix a, Matrix b)
		{
			if (a.Cols != b.Rows)
			{
				throw new Exception("Tried to multiply unmultiplicable matrices in Matrix.Multiply!");
			}
			Matrix res = new Matrix(a.Rows, b.Cols);
			for (int i = 0; i < res.Rows; i++)
			{
				for (int j = 0; j < res.Cols; j++)
				{
					res[i, j] = 0d;
					for (int k = 0; k < a.Cols; k++)
					{
						res[i, j] += a[i, k] * b[k, j];
					}
				}
			}
			return res;
		}

		public static Matrix Subtract(Matrix a, Matrix b)
		{
			if (a.Rows != b.Rows || a.Cols != b.Cols)
			{
				throw new Exception("ERROR: Tried to subtract matrixes of different size!");
			}
			Matrix res = new Matrix(a.Rows, a.Cols);
			for (int i = 0; i < a.Rows; i++)
			{
				for (int j = 0; j < a.Cols; j++)
				{
					res[i, j] = a[i, j] - b[i, j];
				}
			}
			return res;
		}

		public static Matrix Subtract(Matrix m, double n)
		{
			Matrix res = new Matrix(m.Rows, m.Cols);
			for (int i = 0; i < m.Rows; i++)
			{
				for (int j = 0; j < m.Cols; j++)
				{
					res[i, j] = m[i, j] - n;
				}
			}
			return res;
		}

		public static Matrix Add(Matrix a, Matrix b)
		{
			if (a.Rows != b.Rows || a.Cols != b.Cols)
			{
				throw new Exception("ERROR: Tried to add matrixes of different size!");
			}
			Matrix res = new Matrix(a.Rows, a.Cols);
			for (int i = 0; i < a.Rows; i++)
			{
				for (int j = 0; j < a.Cols; j++)
				{
					res[i, j] = a[i, j] + b[i, j];
				}
			}
			return res;
		}

		public static Matrix Add(Matrix m, double n)
		{
			Matrix res = new Matrix(m.Rows, m.Cols);
			for (int i = 0; i < m.Rows; i++)
			{
				for (int j = 0; j < m.Cols; j++)
				{
					res[i, j] = m[i, j] + n;
				}
			}
			return res;
		}

		public static Matrix Scale(Matrix m, double n)
		{
			Matrix res = new Matrix(m.Rows, m.Cols);
			for (int i = 0; i < m.Rows; i++)
			{
				for (int j = 0; j < m.Cols; j++)
				{
					res[i, j] = m[i, j] * n;
				}
			}
			return res;
		}

		public static Matrix Evaluate(Matrix m, Func<double, double> mapper)
		{
			Matrix res = new Matrix(m.Rows, m.Cols);
			for (int i = 0; i < m.Rows; i++)
			{
				for (int j = 0; j < m.Cols; j++)
				{
					res.data[i, j] = mapper(m.data[i, j]);
				}
			}
			return res;
		}

		public static Matrix Transpose(Matrix m)
		{
			Matrix res = new Matrix(m.Cols, m.Rows);
			for (int i = 0; i < m.Rows; i++)
			{
				for (int j = 0; j < m.Cols; j++)
				{
					res[j, i] = m[i, j];
				}
			}
			return res;
		}
	}
}
