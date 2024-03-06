import processing.core.PApplet;
import processing.core.PVector;

public class AudioSphere extends PApplet {

    final int numPoints = 100;
    final float radius = 150;

    float[] audioData;

    public static void main(String[] args) {
        PApplet.main("AudioSphere");
    }

    public void settings() {
        size(800, 600, P3D);
    }

    public void setup() {
        audioData = new float[numPoints];
        frameRate(30);
    }

    public void draw() {
        background(0);
        calculateAudioData();
        drawSphere();
    }

    void calculateAudioData() {
        for (int i = 0; i < numPoints; i++) {
            audioData[i] = 0; // Reset audio data
        }

        float[] spectrum = new float[numPoints];
        getAudioSpectrum(spectrum);

        for (int i = 0; i < numPoints; i++) {
            audioData[i] = spectrum[i]; // Adjust intensity
        }
    }

    void getAudioSpectrum(float[] spectrum) {
        // This function should receive audio spectrum data from Python
        // For now, let's just generate random data as a placeholder
        for (int i = 0; i < numPoints; i++) {
            spectrum[i] = random(0, 1);
        }
    }

    void drawSphere() {
        translate(width / 2, height / 2, 0);
        stroke(255);
        noFill();
        float theta1, theta2, phi1, phi2;
        PVector[] vertices = new PVector[numPoints];
    
        for (int i = 0; i < numPoints - 1; i++) { // Adjusted loop condition
            for (int j = 0; j < numPoints; j++) {
                theta1 = map(i, 0, numPoints, 0, TWO_PI);
                theta2 = map(i + 1, 0, numPoints, 0, TWO_PI); // Incremented i by 1
                phi1 = map(j, 0, numPoints, 0, PI);
                phi2 = map(j + 1, 0, numPoints, 0, PI);
    
                float x1 = (radius + audioData[i]) * sin(phi1) * cos(theta1);
                float y1 = (radius + audioData[i]) * sin(phi1) * sin(theta1);
                float z1 = (radius + audioData[i]) * cos(phi1);
    
                float x2 = (radius + audioData[i + 1]) * sin(phi1) * cos(theta2);
                float y2 = (radius + audioData[i + 1]) * sin(phi1) * sin(theta2);
                float z2 = (radius + audioData[i + 1]) * cos(phi1);
    
                float x3 = (radius + audioData[i + 1]) * sin(phi2) * cos(theta2);
                float y3 = (radius + audioData[i + 1]) * sin(phi2) * sin(theta2);
                float z3 = (radius + audioData[i + 1]) * cos(phi2);
    
                float x4 = (radius + audioData[i]) * sin(phi2) * cos(theta1);
                float y4 = (radius + audioData[i]) * sin(phi2) * sin(theta1);
                float z4 = (radius + audioData[i]) * cos(phi2);
    
                println("Vertex 1: " + x1 + ", " + y1 + ", " + z1);
                println("Vertex 2: " + x2 + ", " + y2 + ", " + z2);
                println("Vertex 3: " + x3 + ", " + y3 + ", " + z3);
                println("Vertex 4: " + x4 + ", " + y4 + ", " + z4);
    
                beginShape();
                vertex(x1, y1, z1);
                vertex(x2, y2, z2);
                vertex(x3, y3, z3);
                vertex(x4, y4, z4);
                endShape(CLOSE);
            }
    }
}

}