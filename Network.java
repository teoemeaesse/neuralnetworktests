import java.util.Random;

public class Network {
    private float[][][] weights; // LAYERS x NEURONS x INCOMING SYNAPSE WEIGHTS
    private float[][] biases;    // LAYERS x NEURONS
    private int[] layers;

    public Network(int[] layers){
        this.layers = layers;
        generateBiases();
        generateWeights();
    }

    public float train(float[] inputs, float[] expected){
        inputs = inputs.clone();
        expected = expected.clone();
        float[][] weightedSums = getWeightedSums(inputs);

        float averageCost = 0;
        for(int y = 0; y < expected.length; y++)
            averageCost += Math.pow(sigmoid(weightedSums[weightedSums.length-1][y]) - expected[y], 2);
        averageCost /= expected.length;

        float[][] delta = new float[weights.length][];
        delta[delta.length-1] = new float[expected.length];
        for(int y = 0; y < expected.length; y++)
            delta[delta.length-1][y] = 2 * (sigmoid(weightedSums[weightedSums.length-1][y]) - expected[y]);

        for(int l = delta.length-1; l > 0; l--){
            delta[l-1] = new float[weightedSums[l].length];
            for(int y = 0; y < weightedSums[l].length; y++){
                float averageDelta = 0;
                for(int n = 0; n < weightedSums[l+1].length; n++)
                    averageDelta += weights[l][n][y] * sigmoidPrime(weightedSums[l+1][n]) * delta[l][n];
                delta[l-1][y] = averageDelta / delta[l].length;
            }
        }

        for(int l = 0; l < weights.length; l++){
            for(int n = 0; n < weights[l].length; n++){
                float dadz = sigmoidPrime(weightedSums[l+1][n]), dcda = delta[l][n];
                for(int w = 0; w < weights[l][n].length; w++)
                    weights[l][n][w] -= sigmoid(weightedSums[l][w]) * dadz * dcda;
                biases[l+1][n] -= dadz * dcda;
            }
        }

        return averageCost;
    }

    public float[][] getWeightedSums(float[] inputs){
        float[][] activations = new float[biases.length][];
        for(int l = 0; l < biases.length; l++)
            activations[l] = new float[biases[l].length];

        activations[0] = inputs;
        for(int l = 0; l < weights.length; l++){
            for(int n = 0; n < weights[l].length; n++){
                float weightedSum = 0;
                for(int w = 0; w < weights[l][n].length; w++)
                    weightedSum += weights[l][n][w] * sigmoid(activations[l][w]);
                activations[l+1][n] = weightedSum + biases[l+1][n];
            }
        }
        return activations;
    }

    private void generateBiases(){
        biases = new float[layers.length][];
        for(int l = 0; l < layers.length; l++){
            biases[l] = new float[layers[l]];
            for(int n = 0; n < layers[l]; n++)
                biases[l][n] = new Random().nextFloat() * 2 - 1;
        }
    }

    private void generateWeights(){
        weights = new float[layers.length-1][][];
        for(int l = 0; l < layers.length-1; l++){
            weights[l] = new float[layers[l+1]][];
            for(int n = 0; n < layers[l+1]; n++){
                weights[l][n] = new float[layers[l]];
                for(int w = 0; w < layers[l]; w++)
                    weights[l][n][w] = new Random().nextFloat() * 2 - 1;
            }
        }
    }

    private static float sigmoid(float x){
        if(x < -10)
            return 0;
        else if(x > 10)
            return 1;
        return 1f / (1 + (float) Math.exp(-x));
    }

    private static float sigmoidPrime(float x){
        if(x < -10 || x > 10)
            return 0;
        float exp = (float) Math.exp(-x);
        return exp / ((1 + exp) * (1 + exp));
    }
}
