using Smoke;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Diagnostics;
using System.Net;
using System.Xml;

namespace Smoke
{
    public class Cost
    {
        public static float[] CrossEntropy(float[][] Predictions, float[][] Y_Labels)
        {
            for (int i=0; i<Predictions.Length; i++)
            {
                Predictions[i] = NNMath.Subtract(Predictions[i], Y_Labels[i]);
            }
            for (int i=0; i<Predictions.Length-1;i++)
            {
                Predictions[0] = NNMath.VecAdd(Predictions[0], Predictions[i + 1]);
            }
            float[] Output = NNMath.ScalarDivision(Predictions[0], Predictions.Length);


            return Output;
        }
    }

    public class Utils
    {
        public static float[] GetColumn(float[][] Matrix, int Dim)
        {
            float[] output = new float[Matrix.GetLength(0)];
            for (int i=0; i<Matrix.GetLength(0); i++)
            {
                output[i] = Matrix[i][Dim];
            }


            return output;
        }
        public static float[][] CopyMatrix(float[][] Matrix)
        {
            float[][] output = new float[Matrix.Length][];
            for (int i=0; i< Matrix.Length;i++)
            {
                output[i] = new float[Matrix[0].Length];
                for (int j=0;j<Matrix[0].Length;j++)
                {
                    output[i][j] = Matrix[i][j];
                }
            }
            return output;
        }

        public static float[][] Zeros(int Dim1, int Dim2)
        {
            float[][] output = new float[Dim1][];
            for (int i=0;i<Dim1;i++)
            {
                output[i] = new float[Dim2];
                for (int j=0; j<Dim2;j++)
                {
                    output[i][j] = 0;
                }
            }
            return output;
        }
        public static float[] Zeros(int Dim)
        {
            float[] output = new float[Dim];
            for (int i=0; i<Dim;i++)
            {
                output[i] = 0;
            }
            return output;
        }
    }

    public class NNRand
    {
        public static float[][] NormalSample(float mean, float stdDev, int Xdim, int Ydim)
        {
            //random sampling from a gausian distribution
            int numSamples = Xdim * Ydim;

            float[][] output = new float[Xdim][];

            for (int i = 0; i < Xdim; i++)
            {
                output[i] = new float[Ydim];
                for (int j = 0; j < Ydim; j++)
                {
                    Random rand = new Random();
                    double u1 = 1.0 - rand.NextDouble();
                    double u2 = 1.0 - rand.NextDouble();
                    double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) *
                    Math.Sin(2.0 * Math.PI * u2);
                    double randNormal =
                        mean + stdDev * randStdNormal;

                    output[i][j] = (float)randNormal;
                }
            }
            return output;
        }


        public static float[,] NormalSample(float mean, float stdDev, int[] dims)
        {
            //random sampling from a gausian distribution
            int numSamples = 0;
            for (int i=0;i<dims.Length;i++)
            {
                numSamples += dims[i];
            }

            float[,] output = new float[dims[0], dims[1]];
            
            for (int i=0; i < dims[0]; i++)
            {
                for (int j = 0; j < dims[1]; j++)
                {
                    Random rand = new Random();
                    double u1 = 1.0 - rand.NextDouble();
                    double u2 = 1.0 - rand.NextDouble();
                    double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) *
                    Math.Sin(2.0 * Math.PI * u2);
                    double randNormal =
                        mean + stdDev * randStdNormal;

                    output[i,j] = (float)randNormal;
                }
            }
            return output;
        }
    }

    public class NNMath
    {
        public static float[][] Transpose(float[][] Matrix)
        {
            float[][] output = new float[Matrix[0].Length][];
            for (int i=0; i<Matrix[0].Length; i++)
            {
                output[i] = new float[Matrix.Length];
                for (int j=0; j<Matrix.Length;j++)
                {
                    output[i][j] = Matrix[j][i];
                }
            }
            return output;
        }

        public static float[] ScalarDivision(float[] v1, float scalar)
        {
            for (int i=0; i<v1.Length;i++)
            {
                v1[i] = v1[i] / scalar;
            }
            return v1;
        }

        public static float[] Subtract(float[] v1, float[] v2)
        {
            Trace.Assert(v1.Length == v2.Length, "Vectors must be of equal length for subtraction");
            for (int i=0; i< v1.Length; i++)
            {
                v1[i] = v1[i] - v2[i];
            }
            return v1;
        }

        public static float Exp(float X)
        {
            return (float)Math.Pow(Math.E, X);
        }

        public static float Sum(float[] X)
        {
            float output = 0;
            for (int i = 0; i < X.Length; i++)
            {
                output += X[i];
            }
            return output;
        }

        public static float Dot(float[] Vector1, float[] Vector2)
        {
            Trace.Assert(Vector1.Length == Vector2.Length, "Vector lengths do not match");

            float result = 0;
            for (int i = 0; i < Vector1.Length; i++)
            {
                result += Vector1[i] * Vector2[i];
            }
            return result;
        }

        public static float[][] MatAdd(float[][] Matrix1, float[][] Matrix2)
        {
            Trace.Assert(Matrix1.Length == Matrix2.Length && Matrix1[0].Length == Matrix2[0].Length,
                "Matrices must have the same shape");

            for (int i = 0; i < Matrix1.Length; i++)
            {
                for (int j = 0; j < Matrix1[0].Length; j++)
                {
                    Matrix1[i][j] = Matrix1[i][j] + Matrix2[i][j];
                }
            }
            return Matrix1;
        }

        public static float[] VecAdd(float[] Vector1, float[] Vector2)
        {
            Trace.Assert(Vector1.Length == Vector2.Length, "Vectors must have equal length");
            for (int i=0; i<Vector1.Length; i++)
            {
                Vector1[i] += Vector2[i];
            }
            return Vector1;
        }

        public static float[] MatAdd(float[] Vector1, float[] Vector2)
        {
            for (int i=0; i< Vector1.Length;i++)
            {
                Vector1[i] += Vector2[i];
            }
            return Vector1;
        }


        public static float[][] MatMul(float[][] Matrix1, float[][] Matrix2)
        {
            Trace.Assert(Matrix1.Rank == Matrix2.Rank, "Matrices must be of equal rank");

            //new Matrix is of size Rows (from Matrix1) x Columns (From Matrix 2)
            float[][] output = new float[Matrix1.GetLength(0)][];
            for (int i=0; i<Matrix1.GetLength(0);i++)
            {
                output[i] = new float[Matrix2[0].Length];
            }
            
            for(int i=0; i<Matrix1.GetLength(0); i++)
            {
                for (int j=0; j<Matrix2[0].Length; j++)
                {
                    //output[i][j] = NNMath.Dot(Matrix1[i], Utils.GetColumn(Matrix2, j));
                    for (int k=0; k<Matrix2.Length; k++)
                    {
                        output[i][j] = Matrix1[i][k] * Matrix2[k][j];
                    }
                }
            }

            return output;

        }
    }

    public class Activation
    {

        public static float[] Sigmoid(float[] X)
        {
            for (int i=0; i<X.Length; i++)
            {
                X[i] = 1 / (1 + NNMath.Exp(X[i]));
            }
            return X;

        }

        public static float[] Tanh(float[] X)
        {
            for (int i=0; i<X.Length;i++)
            {
                X[i] = (NNMath.Exp(X[i]) - NNMath.Exp(-X[i])) / (NNMath.Exp(X[i]) + NNMath.Exp(-X[i]));
            }
            return X;
        }

        public static float[] Softmax(float[] X)
        {
            for (int i = 0; i < X.Length; i++)
            {
                X[i] = NNMath.Exp(X[i]);

            }
            var expXSum = NNMath.Sum(X);

            for (int i = 0; i < X.Length; i++)
            {
                X[i] = X[i] / expXSum;
            }

            return X;
        }

        public static float[] TanhDerivative(float[] X)
        {
            for (int i=0; i < X.Length; i++)
            {
                X[i] = 1 - (float)Math.Pow(X[i], 2);
            }
            return X;
        }

    }
    public class LinearLayer
    {
        public int InputSize;
        public int OutputSize;

        private float[][] Weights;
        private float[] Bias;

        public float[][] Z;
        public float[][] A;

        private String Activ;
        public LinearLayer(int InputDim, int OutputDim, String Activation="None")
        {
            Activ = Activation;
            InputSize = InputDim;
            OutputSize = OutputDim;
            Weights = NNRand.NormalSample(0, (float)0.1, InputDim, OutputDim);
            Bias = Utils.Zeros(OutputDim);

            A = new float[InputDim][];

        }

        public float[][] Forward(float[][] X)
        {
            //Multiply inputs with the weights
            float[][] Output = NNMath.MatMul(X, Weights);
            //Add the bias
            for (int i=0;i<Output.Length;i++)
            {
                Output[i] = NNMath.MatAdd(Output[i], Bias);
            }

            //Apply activation function
            if (Activ == "sigmoid")
            {
                for (int i=0; i<X.Length;i++)
                {
                   Output[i] = Activation.Sigmoid(Output[i]);
                }
            }
            A = Utils.CopyMatrix(Output);
            Console.WriteLine(A[0][0]);
            return Output;
        }

    }

    public class Net
    {
        private List<LinearLayer> LinearLayers = new List<LinearLayer>();

        public void AddLayer(int InputDim, int OutputDim, String Activation="None")
        {
            if (LinearLayers.Count != 0)
            {
                //Make sure layer being added is valid
                int LastOutputDim = LinearLayers[LinearLayers.Count - 1].OutputSize;
                Trace.Assert(LastOutputDim == InputDim, "New layer input dimension must match the output dimension of the previous layer");
            }

            LinearLayer Layer = new LinearLayer(InputDim, OutputDim, Activation);
            LinearLayers.Add(Layer);
        }

        public float[][] Forward(float[][] X)
        {
            for (int i=0; i<LinearLayers.Count;i++)
            {
                X = LinearLayers[i].Forward(X);
            }
            return X;
        }


    }


    public class LSTM
    {
        private int InputSize;
        private int HiddenSize;
        private int TotalInputWeightSize;
        private int OutputSize;

        public LSTM(int InputDim, int HiddenDim, int OutputDim)
        {
            InputSize = InputDim;
            HiddenSize = HiddenDim;
            TotalInputWeightSize = InputDim + HiddenDim;
            OutputSize = OutputDim;
        }

        public Dictionary<String, float[,]> InitializeParameters()
        {
            Dictionary<String, float[,]> parameters = new Dictionary<string, float[,]>();

            float mean = 0;
            float std = (float) 0.01;
            int[] dim = { InputSize + HiddenSize, HiddenSize };
            float[,] forget_gate_weights = NNRand.NormalSample(mean, std, dim);

            float[,] input_gate_weights = NNRand.NormalSample(mean, std, dim);

            float[,] output_gate_weights = NNRand.NormalSample(mean, std, dim);

            float[,] gate_gate_units = NNRand.NormalSample(mean, std, dim);

            dim[0] = HiddenSize;
            dim[1] = OutputSize;
            float[,] hidden_output_weights = NNRand.NormalSample(mean, std, dim);

            parameters["fgw"] = forget_gate_weights;
            parameters["igw"] = input_gate_weights;
            parameters["ogw"] = output_gate_weights;
            parameters["ggw"] = gate_gate_units;
            parameters["how"] = hidden_output_weights;

            return parameters;

        }

        
    }


}
