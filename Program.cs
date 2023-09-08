using Microsoft.ML.Data;
using Microsoft.ML;
using System;
using System.Security.Cryptography.X509Certificates;

Console.WriteLine("Hello, World!");

namespace WineQualityPrediction
{
    public class WineData
    {
        // Data columns from the CSV
        public float FixedAcidity { get; set; }
        public float VolatileAcidity { get; set; }
        public float CitricAcid { get; set; }
        public float ResidualSugar { get; set; }
        public float Chlorides { get; set; }
        public int FreeSulfurDioxide { get; set; }
        public int TotalSulfurDioxide { get; set; }
        public float Density { get; set; }
        public float PH { get; set; }
        public float Sulphates { get; set; }
        public float Alcohol { get; set; }
        public int Quality { get; set; } // Target column
    }

    public class WinePrediction
    {
        [ColumnName("Score")]
        public float Quality;
    }

    class Program
    {
        static void Main(string[] args)
        {
            var context = new MLContext();

            // Load data
            var data = context.Data.LoadFromTextFile<WineData>("./winequality-red.csv", separatorChar: ';');

            // Split the data
            var trainTestData = context.Data.TrainTestSplit(data, testFraction: 0.2);

            // Define pipeline
            var pipeline = context.Transforms.Concatenate("Features", nameof(WineData.FixedAcidity) /*, Other columns*/)
                            .Append(context.Regression.Trainers.Sdca(labelColumnName: nameof(WineData.Quality), maximumNumberOfIterations: 100));

            // Train the model
            var model = pipeline.Fit(trainTestData.TrainSet);

            // Evaluate
            var predictions = model.Transform(trainTestData.TestSet);
            var metrics = context.Regression.Evaluate(predictions, labelColumnName: nameof(WineData.Quality));

            Console.WriteLine($"R^2: {metrics.RSquared}");
            // ... Other metrics

            // Predict a sample wine's quality
            var sampleWine = new WineData() { FixedAcidity = 7.4f /*, Other values*/ };
            var size = new[] { sampleWine };
            var sizeNew = context.Data.LoadFromEnumerable(size);
            var prediction = model.Transform(sizeNew);
            var sizePredicted = context.Data.CreateEnumerable<WinePrediction>(prediction, reuseRowObject: false);

            Console.WriteLine($"Predicted Quality: {sizePredicted.Quality}");
        }
    }
}
