using Microsoft.ML.Data;
using Microsoft.ML;
using System;
using System.Security.Cryptography.X509Certificates;
using System.Globalization;

namespace WineQualityPrediction
{
    public class WineData
    {
        [LoadColumn(0)]
        public float FixedAcidity { get; set; }

        [LoadColumn(1)]
        public float VolatileAcidity { get; set; }

        [LoadColumn(2)]
        public float CitricAcid { get; set; }

        [LoadColumn(3)]
        public float ResidualSugar { get; set; }

        [LoadColumn(4)]
        public float Chlorides { get; set; }

        [LoadColumn(5)]
        public float FreeSulfurDioxide { get; set; }

        [LoadColumn(6)]
        public float TotalSulfurDioxide { get; set; }

        [LoadColumn(7)]
        public float Density { get; set; }

        [LoadColumn(8)]
        public float PH { get; set; }

        [LoadColumn(9)]
        public float Sulphates { get; set; }

        [LoadColumn(10)]
        public float Alcohol { get; set; }

        [LoadColumn(11)]
        public float Quality { get; set; }

        [NoColumn] //We do this so it doesnt try read it from the csv.
        public string WineType { get; set; }

        // Add a default constructor
        public WineData() { }

        public WineData(string wineType = "red") //use a constructor so we can set it when loading data.
        {
            WineType = wineType;
        }

    }


    public class WinePrediction
    {
        [ColumnName("Score")]
        public float Quality { get; set; }
    }

    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Hello, World!");

            Console.WriteLine("Creating Context");
            var context = new MLContext();

            Console.WriteLine("Working Directory: " + Environment.CurrentDirectory);

            var whiteWines = LoadWineData("../../../winequality-white.csv", "White");
            var redWines = LoadWineData("../../../winequality-red.csv", "Red");

            static List<WineData> LoadWineData(string filePath, string wineType)
            {
                List<WineData> wines = new List<WineData>();
                using (var reader = new StreamReader(filePath))
                {
                    reader.ReadLine(); // skip header
                    while (!reader.EndOfStream)
                    {
                        var line = reader.ReadLine();
                        var values = line.Split(';');

                        var wine = new WineData
                        {
                            FixedAcidity = float.Parse(values[0], CultureInfo.InvariantCulture),
                            VolatileAcidity = float.Parse(values[1], CultureInfo.InvariantCulture),
                            CitricAcid = float.Parse(values[2], CultureInfo.InvariantCulture),
                            ResidualSugar = float.Parse(values[3], CultureInfo.InvariantCulture),
                            Chlorides = float.Parse(values[4], CultureInfo.InvariantCulture),
                            FreeSulfurDioxide = float.Parse(values[5], CultureInfo.InvariantCulture),
                            TotalSulfurDioxide = float.Parse(values[6], CultureInfo.InvariantCulture),
                            Density = float.Parse(values[7], CultureInfo.InvariantCulture),
                            PH = float.Parse(values[8], CultureInfo.InvariantCulture),
                            Sulphates = float.Parse(values[9], CultureInfo.InvariantCulture),
                            Alcohol = float.Parse(values[10], CultureInfo.InvariantCulture),
                            Quality = float.Parse(values[11], CultureInfo.InvariantCulture),
                            WineType = wineType
                        };
                        wines.Add(wine);
                    }
                }
                return wines;
            }

            Console.WriteLine("Combine Data");
            var allWines = whiteWines.Concat(redWines);
            var data = context.Data.LoadFromEnumerable(allWines);
            data = context.Data.ShuffleRows(data);


            // Split the data
            Console.WriteLine("Splitting Data");
            var trainTestData = context.Data.TrainTestSplit(data, testFraction: 0.2);

            // Define pipeline
            Console.WriteLine("Defining Pipeline");
            var pipeline = context.Transforms.Conversion.ConvertType("FreeSulfurDioxide", "FreeSulfurDioxide", DataKind.Single)
               .Append(context.Transforms.Conversion.ConvertType("TotalSulfurDioxide", "TotalSulfurDioxide", DataKind.Single))
               .Append(context.Transforms.Concatenate("Features",
                nameof(WineData.FixedAcidity), nameof(WineData.CitricAcid), nameof(WineData.VolatileAcidity), nameof(WineData.ResidualSugar),
                nameof(WineData.Chlorides), nameof(WineData.FreeSulfurDioxide), nameof(WineData.TotalSulfurDioxide), nameof(WineData.Density),
                nameof(WineData.PH), nameof(WineData.Sulphates), nameof(WineData.Alcohol))
             .Append(context.Regression.Trainers.Sdca(labelColumnName: nameof(WineData.Quality), maximumNumberOfIterations: 100)));

            // Train the model
            Console.WriteLine("Training Model");
            var model = pipeline.Fit(trainTestData.TrainSet);

            // Evaluate
            Console.WriteLine("Transforming Prediction on test");
            var predictions = model.Transform(trainTestData.TestSet);

            Console.WriteLine("Evaluating Prediction");
            var metrics = context.Regression.Evaluate(predictions, labelColumnName: nameof(WineData.Quality));

            Console.WriteLine("Closer to 1 = Better fitted model");
            Console.WriteLine($"R^2: {metrics.RSquared}");
            Console.WriteLine($"Mean Absolute Error: {metrics.MeanAbsoluteError}");
            Console.WriteLine($"Mean Squared Error: {metrics.MeanSquaredError}");
            Console.WriteLine($"Root Mean Squared Error: {metrics.RootMeanSquaredError}");

            // Predict a sample wine's quality
            Console.WriteLine("Creating sample to predict quality");
            var sampleWine = new WineData()
            {
                FixedAcidity = 7.4f,
                VolatileAcidity = 2.4f,
                CitricAcid = 1.4f,
                ResidualSugar = 4.4f,
                Chlorides = 1.4f,
                FreeSulfurDioxide = 1,
                TotalSulfurDioxide = 2,
                Density = 1.4f,
                PH = 7.4f,
                Sulphates = 3.4f,
                Alcohol = 12.4f,
                WineType = "red"
            };
            var size = new[] { sampleWine };
            var sizeNew = context.Data.LoadFromEnumerable(size);
            var prediction = model.Transform(sizeNew);
            var sizePredicted = context.Data.CreateEnumerable<WinePrediction>(prediction, reuseRowObject: false);

            var predictedQuality = sizePredicted.First().Quality;
            Console.WriteLine($"Predicted Quality: {predictedQuality}");

        }
    }
}