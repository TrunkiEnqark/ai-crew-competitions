use std::error::Error;
use std::fs::File;
use csv::ReaderBuilder;
use rand::prelude::*;

#[derive(Debug, Clone)]
struct Sample {
    label: u8,
    pixels: Vec<f32>,
}

struct KNNClassifier {
    train_data: Vec<Sample>,
    k: usize,
}

impl KNNClassifier {
    fn new(k: usize) -> Self {
        KNNClassifier {
            train_data: Vec::new(),
            k,
        }
    }

    fn fit(&mut self, samples: Vec<Sample>) {
        self.train_data = samples;
    }

    fn euclidean_distance(&self, a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f32>()
            .sqrt()
    }

    fn predict(&self, pixels: &[f32]) -> u8 {
        let mut distances: Vec<(f32, u8)> = self
            .train_data
            .iter()
            .map(|sample| {
                let dist = self.euclidean_distance(&sample.pixels, pixels);
                (dist, sample.label)
            })
            .collect();

        distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        
        let k_nearest = &distances[..self.k];
        
        let mut label_counts = vec![0; 10];
        for (_, label) in k_nearest {
            label_counts[*label as usize] += 1;
        }

        label_counts
            .iter()
            .enumerate()
            .max_by_key(|(_, &count)| count)
            .map(|(label, _)| label as u8)
            .unwrap()
    }
}

fn load_training_data(path: &str) -> Result<Vec<Sample>, Box<dyn Error>> {
    let file = File::open(path)?;
    let mut rdr = ReaderBuilder::new().has_headers(true).from_reader(file);
    
    let mut samples = Vec::new();
    
    for result in rdr.records() {
        let record = result?;
        let values: Vec<f32> = record.iter()
            .map(|s| s.parse::<f32>().unwrap())
            .collect();
        
        let label = values[0] as u8;
        let pixels: Vec<f32> = values[1..].iter().map(|&x| x / 255.0).collect();
            
        samples.push(Sample { label, pixels });
    }
    
    Ok(samples)
}

fn load_test_data(path: &str) -> Result<Vec<Vec<f32>>, Box<dyn Error>> {
    let file = File::open(path)?;
    let mut rdr = ReaderBuilder::new().has_headers(true).from_reader(file);
    
    let mut samples = Vec::new();
    
    for result in rdr.records() {
        let record = result?;
        let pixels: Vec<f32> = record.iter()
            .map(|s| s.parse::<f32>().unwrap() / 255.0)
            .collect();
            
        samples.push(pixels);
    }
    
    Ok(samples)
}

fn write_predictions(predictions: &[u8], output_path: &str) -> Result<(), Box<dyn Error>> {
    let mut wtr = csv::Writer::from_path(output_path)?;
    
    wtr.write_record(&["ImageId", "Label"])?;
    
    for (i, &pred) in predictions.iter().enumerate() {
        wtr.write_record(&[(i + 1).to_string(), pred.to_string()])?;
    }
    
    wtr.flush()?;
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    println!("Loading training data...");
    let mut samples = load_training_data("data/train.csv")?;
    
    let mut rng = thread_rng();
    samples.shuffle(&mut rng);
    
    let training_size = 28000;
    let training_samples = samples[..training_size].to_vec();
    
    println!("Training KNN classifier...");
    let mut classifier = KNNClassifier::new(3);
    classifier.fit(training_samples);
    
    println!("Loading test data...");
    let test_data = load_test_data("data/test.csv")?;
    
    println!("Making predictions...");
    let mut predictions = Vec::new();
    for (i, pixels) in test_data.iter().enumerate() {
        if i % 1000 == 0 {
            println!("Processing image {}/{}", i, test_data.len());
        }
        let prediction = classifier.predict(pixels);
        predictions.push(prediction);
    }
    
    println!("Writing predictions to output.csv...");
    write_predictions(&predictions, "data/output.csv")?;
    
    println!("Done!");
    Ok(())
}