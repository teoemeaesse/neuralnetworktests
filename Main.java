import java.util.Random;

public class Main {
    private static final int INPUT_COUNT = 5000, EPOCHS = 10000000;

    public static void main(String[] args){
        Network network = new Network(new int[]{2, 5, 10});
        float[][] inputs = new float[INPUT_COUNT][2],
                expected = new float[INPUT_COUNT][10];
        for(int i = 0; i < INPUT_COUNT; i++){
            inputs[i][0] = new Random().nextFloat();
            inputs[i][1] = new Random().nextFloat();
            int res = (int) (((inputs[i][0] + inputs[i][1]) / 2) * 10);
            for(int j = 0; j < expected[i].length; j++){
                if(j == res){
                    expected[i][j] = 1;
                }else{
                    expected[i][j] = 0;
                }
            }
        }
        //for(int i = 0; i < EPOCHS; i++){
        int i = 0;
        float averageCost = 0;
        while(true){
            i++;
            int mb = new Random().nextInt(inputs.length);
            if(i % 100000 == 0){
                System.out.println("Average cost for last 100000 epochs: " + averageCost / 100000f);
                averageCost = 0;
            }
            else averageCost += network.train(inputs[mb], expected[mb]);
        }
    }
}
