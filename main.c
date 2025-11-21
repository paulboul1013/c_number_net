#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <SDL2/SDL.h>


// Model filename
#define MODEL_FILENAME "model.bin"

// ==========================================
// CSV Reading Tools
// ==========================================
typedef struct {
    int rows;
    int cols;
    int **data;
} CSVData;

CSVData* read_csv(const char* filename);
void free_csv(CSVData* csv);

CSVData* read_csv(const char* filename) {
    FILE* file = fopen(filename, "r");
    if (!file) { printf("Cannot open file: %s\n", filename); return NULL; }

    CSVData* csv = (CSVData*)malloc(sizeof(CSVData));
    char line[100000];
    int row_count = 0, capacity = 1000;
    
    csv->data = (int**)malloc(capacity * sizeof(int*));
    
    // Read title line and calculate number of columns
    if (fgets(line, sizeof(line), file)) {
        int col_count = 1;
        for (char* p = line; *p; p++) if (*p == ',') col_count++;
        csv->cols = col_count;
    }

    // Read data lines
    while (fgets(line, sizeof(line), file)) {
        // If capacity needs extension
        if (row_count >= capacity) {
            capacity *= 2;
            csv->data = (int**)realloc(csv->data, capacity * sizeof(int*));
        }
        csv->data[row_count] = (int*)malloc(csv->cols * sizeof(int));
        
        char* token = strtok(line, ",");
        int col = 0;
        while (token && col < csv->cols) {
            csv->data[row_count][col++] = atoi(token);
            token = strtok(NULL, ",");
        }
        row_count++;
    }
    csv->rows = row_count;
    fclose(file);
    return csv;
}

void free_csv(CSVData* csv) {
    if (csv) {
        for (int i = 0; i < csv->rows; i++) free(csv->data[i]);
        free(csv->data);
        free(csv);
    }
}

// ==========================================
// Neural Network Design
// ==========================================

#define INPUT_NODES 784
#define HIDDEN_NODES 128
#define OUTPUT_NODES 10
#define LEARNING_RATE 0.1

typedef struct {
    double *hidden_weights; // Input -> Hidden weights (784 * 128)
    double *output_weights; // Hidden -> Output weights (128 * 10)
    double *hidden_bias;    // Hidden layer bias (128)
    double *output_bias;    // Output layer bias (10)
} NeuralNetwork;

// Math helper functions
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double d_sigmoid(double x) {
    return x * (1.0 - x); // Assume x is already the output of sigmoid
}

// Initialize memory and random weights
void init_network(NeuralNetwork* nn) {
    nn->hidden_weights = (double*)malloc(INPUT_NODES * HIDDEN_NODES * sizeof(double));
    nn->output_weights = (double*)malloc(HIDDEN_NODES * OUTPUT_NODES * sizeof(double));
    nn->hidden_bias = (double*)malloc(HIDDEN_NODES * sizeof(double));
    nn->output_bias = (double*)malloc(OUTPUT_NODES * sizeof(double));

    // Random initialization (-0.5 to 0.5)
    for (int i = 0; i < INPUT_NODES * HIDDEN_NODES; i++) 
        nn->hidden_weights[i] = ((double)rand() / RAND_MAX) - 0.5;
    
    for (int i = 0; i < HIDDEN_NODES * OUTPUT_NODES; i++) 
        nn->output_weights[i] = ((double)rand() / RAND_MAX) - 0.5;

    for (int i = 0; i < HIDDEN_NODES; i++) nn->hidden_bias[i] = 0.0;
    for (int i = 0; i < OUTPUT_NODES; i++) nn->output_bias[i] = 0.0;
}

// ★ Save model weights to file ★
void save_model(NeuralNetwork* nn, const char* filename) {
    FILE* file = fopen(filename, "wb"); // wb = write binary
    if (!file) {
        printf("Cannot create model file: %s\n", filename);
        return;
    }

    // Write content of four arrays sequentially
    fwrite(nn->hidden_weights, sizeof(double), INPUT_NODES * HIDDEN_NODES, file);
    fwrite(nn->output_weights, sizeof(double), HIDDEN_NODES * OUTPUT_NODES, file);
    fwrite(nn->hidden_bias, sizeof(double), HIDDEN_NODES, file);
    fwrite(nn->output_bias, sizeof(double), OUTPUT_NODES, file);

    fclose(file);
    printf("Model saved to %s\n", filename);
}

//  Load model weights from file 
// Returns 1 for success, 0 for failure
int load_model(NeuralNetwork* nn, const char* filename) {
    FILE* file = fopen(filename, "rb"); // rb = read binary
    if (!file) {
        return 0; // File does not exist
    }

    // Read sequentially (Must match the writing order)
    size_t r1 = fread(nn->hidden_weights, sizeof(double), INPUT_NODES * HIDDEN_NODES, file);
    size_t r2 = fread(nn->output_weights, sizeof(double), HIDDEN_NODES * OUTPUT_NODES, file);
    size_t r3 = fread(nn->hidden_bias, sizeof(double), HIDDEN_NODES, file);
    size_t r4 = fread(nn->output_bias, sizeof(double), OUTPUT_NODES, file);

    fclose(file);

    // Simple check if read size is correct
    if (r1 && r2 && r3 && r4) {
        printf("Model loaded successfully: %s\n", filename);
        return 1;
    } else {
        printf("Model file corrupted or format mismatch.\n");
        return 0;
    }
}

// Training function
void train(NeuralNetwork* nn, double* inputs, double* targets) {
    double hidden_outputs[HIDDEN_NODES];
    double final_outputs[OUTPUT_NODES];

    // --- Forward Pass ---

    // 1. Calculate hidden layer
    for (int j = 0; j < HIDDEN_NODES; j++) {
        double activation = nn->hidden_bias[j];
        for (int k = 0; k < INPUT_NODES; k++) {
            activation += inputs[k] * nn->hidden_weights[k * HIDDEN_NODES + j];
        }
        hidden_outputs[j] = sigmoid(activation);
    }

    // 2. Calculate output layer
    for (int j = 0; j < OUTPUT_NODES; j++) {
        double activation = nn->output_bias[j];
        for (int k = 0; k < HIDDEN_NODES; k++) {
            activation += hidden_outputs[k] * nn->output_weights[k * OUTPUT_NODES + j];
        }
        final_outputs[j] = sigmoid(activation);
    }

    // --- Backward Pass ---

    // 3. Calculate output layer errors & update weights
    double output_errors[OUTPUT_NODES];
    for (int j = 0; j < OUTPUT_NODES; j++) {
        double error = targets[j] - final_outputs[j];
        output_errors[j] = error;
        
        double term = error * d_sigmoid(final_outputs[j]) * LEARNING_RATE;
        for (int k = 0; k < HIDDEN_NODES; k++) {
            nn->output_weights[k * OUTPUT_NODES + j] += term * hidden_outputs[k];
        }
        nn->output_bias[j] += term;
    }

    // 4. Calculate hidden layer errors & update weights
    for (int j = 0; j < HIDDEN_NODES; j++) {
        double error = 0.0;
        for (int k = 0; k < OUTPUT_NODES; k++) {
            error += output_errors[k] * nn->output_weights[j * OUTPUT_NODES + k];
        }
        
        double term = error * d_sigmoid(hidden_outputs[j]) * LEARNING_RATE;
        for (int k = 0; k < INPUT_NODES; k++) {
            nn->hidden_weights[k * HIDDEN_NODES + j] += term * inputs[k];
        }
        nn->hidden_bias[j] += term;
    }
}

// Prediction function
int predict(NeuralNetwork* nn, double* inputs) {
    double hidden_outputs[HIDDEN_NODES];
    double final_outputs[OUTPUT_NODES];

    // Forward pass
    for (int j = 0; j < HIDDEN_NODES; j++) {
        double activation = nn->hidden_bias[j];
        for (int k = 0; k < INPUT_NODES; k++) {
            activation += inputs[k] * nn->hidden_weights[k * HIDDEN_NODES + j];
        }
        hidden_outputs[j] = sigmoid(activation);
    }

    for (int j = 0; j < OUTPUT_NODES; j++) {
        double activation = nn->output_bias[j];
        for (int k = 0; k < HIDDEN_NODES; k++) {
            activation += hidden_outputs[k] * nn->output_weights[k * OUTPUT_NODES + j];
        }
        final_outputs[j] = sigmoid(activation);
    }

    // Find the index of the maximum probability
    int max_index = 0;
    double max_val = final_outputs[0];
    for (int i = 1; i < OUTPUT_NODES; i++) {
        if (final_outputs[i] > max_val) {
            max_val = final_outputs[i];
            max_index = i;
        }
    }
    return max_index;
}

void free_network(NeuralNetwork* nn) {
    free(nn->hidden_weights);
    free(nn->output_weights);
    free(nn->hidden_bias);
    free(nn->output_bias);
}

// ==========================================
// Interactive Drawing Window with SDL2
// ==========================================

// Draw a large digit (0-9) on the window using simple graphics
void draw_digit_in_window(SDL_Renderer* renderer, int digit, int x, int y, int size) {
    // Define 7-segment style patterns for each digit
    // Each digit is drawn as a simple pattern using rectangles
    SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
    
    int thickness = size / 8;
    
    // Draw digit patterns (simplified large display)
    switch (digit) {
        case 0: {
            // Draw a large "0" - circular/oval shape like letter O
            // Draw rounded rectangle border to look like O
            int border_thickness = thickness;
            int padding = size / 6;
            
            // Top horizontal line (curved effect with rounded corners)
            SDL_Rect top = {x + padding, y, size - 2*padding, border_thickness};
            SDL_RenderFillRect(renderer, &top);
            
            // Bottom horizontal line
            SDL_Rect bottom = {x + padding, y + size - border_thickness, size - 2*padding, border_thickness};
            SDL_RenderFillRect(renderer, &bottom);
            
            // Left vertical line (curved inward)
            SDL_Rect left = {x + padding, y + border_thickness, border_thickness, size - 2*border_thickness};
            SDL_RenderFillRect(renderer, &left);
            
            // Right vertical line (curved inward)
            SDL_Rect right = {x + size - padding - border_thickness, y + border_thickness, border_thickness, size - 2*border_thickness};
            SDL_RenderFillRect(renderer, &right);
            
            // Add rounded corners effect
            // Top-left corner
            SDL_Rect tl = {x, y, padding + border_thickness, border_thickness};
            SDL_RenderFillRect(renderer, &tl);
            // Top-right corner
            SDL_Rect tr = {x + size - padding - border_thickness, y, padding + border_thickness, border_thickness};
            SDL_RenderFillRect(renderer, &tr);
            // Bottom-left corner
            SDL_Rect bl = {x, y + size - border_thickness, padding + border_thickness, border_thickness};
            SDL_RenderFillRect(renderer, &bl);
            // Bottom-right corner
            SDL_Rect br = {x + size - padding - border_thickness, y + size - border_thickness, padding + border_thickness, border_thickness};
            SDL_RenderFillRect(renderer, &br);
            break;
        }
        case 1: {
            // Draw "1"
            SDL_Rect r = {x + size/2 - thickness/2, y, thickness, size};
            SDL_RenderFillRect(renderer, &r);
            break;
        }
        case 2: {
            // Draw "2" - top, middle, bottom horizontal bars + right top, left bottom vertical
            SDL_Rect t = {x, y, size, thickness};
            SDL_RenderFillRect(renderer, &t);
            SDL_Rect m = {x, y + size/2 - thickness/2, size, thickness};
            SDL_RenderFillRect(renderer, &m);
            SDL_Rect b = {x, y + size - thickness, size, thickness};
            SDL_RenderFillRect(renderer, &b);
            SDL_Rect rt = {x + size - thickness, y, thickness, size/2};
            SDL_RenderFillRect(renderer, &rt);
            SDL_Rect lb = {x, y + size/2, thickness, size/2};
            SDL_RenderFillRect(renderer, &lb);
            break;
        }
        case 3: {
            // Draw "3" - top, middle, bottom horizontal bars + right vertical
            SDL_Rect t3 = {x, y, size, thickness};
            SDL_RenderFillRect(renderer, &t3);
            SDL_Rect m3 = {x, y + size/2 - thickness/2, size, thickness};
            SDL_RenderFillRect(renderer, &m3);
            SDL_Rect b3 = {x, y + size - thickness, size, thickness};
            SDL_RenderFillRect(renderer, &b3);
            SDL_Rect r3 = {x + size - thickness, y, thickness, size};
            SDL_RenderFillRect(renderer, &r3);
            break;
        }
        case 4: {
            // Draw "4" - left top vertical, right vertical, middle horizontal
            SDL_Rect lt4 = {x, y, thickness, size/2};
            SDL_RenderFillRect(renderer, &lt4);
            SDL_Rect r4 = {x + size - thickness, y, thickness, size};
            SDL_RenderFillRect(renderer, &r4);
            SDL_Rect m4 = {x, y + size/2 - thickness/2, size, thickness};
            SDL_RenderFillRect(renderer, &m4);
            break;
        }
        case 5: {
            // Draw "5" - similar to 2 but mirrored
            SDL_Rect t5 = {x, y, size, thickness};
            SDL_RenderFillRect(renderer, &t5);
            SDL_Rect m5 = {x, y + size/2 - thickness/2, size, thickness};
            SDL_RenderFillRect(renderer, &m5);
            SDL_Rect b5 = {x, y + size - thickness, size, thickness};
            SDL_RenderFillRect(renderer, &b5);
            SDL_Rect lt5 = {x, y, thickness, size/2};
            SDL_RenderFillRect(renderer, &lt5);
            SDL_Rect rb5 = {x + size - thickness, y + size/2, thickness, size/2};
            SDL_RenderFillRect(renderer, &rb5);
            break;
        }
        case 6: {
            // Draw "6" - left vertical, bottom right, all horizontals except top right
            SDL_Rect l6 = {x, y, thickness, size};
            SDL_RenderFillRect(renderer, &l6);
            SDL_Rect m6 = {x, y + size/2 - thickness/2, size, thickness};
            SDL_RenderFillRect(renderer, &m6);
            SDL_Rect b6 = {x, y + size - thickness, size, thickness};
            SDL_RenderFillRect(renderer, &b6);
            SDL_Rect rb6 = {x + size - thickness, y + size/2, thickness, size/2};
            SDL_RenderFillRect(renderer, &rb6);
            break;
        }
        case 7: {
            // Draw "7" - top horizontal + right vertical
            SDL_Rect t7 = {x, y, size, thickness};
            SDL_RenderFillRect(renderer, &t7);
            SDL_Rect r7 = {x + size - thickness, y, thickness, size};
            SDL_RenderFillRect(renderer, &r7);
            break;
        }
        case 8: {
            // Draw "8" - all segments
            SDL_Rect l8 = {x, y, thickness, size};
            SDL_RenderFillRect(renderer, &l8);
            SDL_Rect r8 = {x + size - thickness, y, thickness, size};
            SDL_RenderFillRect(renderer, &r8);
            SDL_Rect t8 = {x, y, size, thickness};
            SDL_RenderFillRect(renderer, &t8);
            SDL_Rect m8 = {x, y + size/2 - thickness/2, size, thickness};
            SDL_RenderFillRect(renderer, &m8);
            SDL_Rect b8 = {x, y + size - thickness, size, thickness};
            SDL_RenderFillRect(renderer, &b8);
            break;
        }
        case 9: {
            // Draw "9" - top, middle horizontal + left top, right verticals
            SDL_Rect t9 = {x, y, size, thickness};
            SDL_RenderFillRect(renderer, &t9);
            SDL_Rect m9 = {x, y + size/2 - thickness/2, size, thickness};
            SDL_RenderFillRect(renderer, &m9);
            SDL_Rect b9 = {x, y + size - thickness, size, thickness};
            SDL_RenderFillRect(renderer, &b9);
            SDL_Rect lt9 = {x, y, thickness, size/2};
            SDL_RenderFillRect(renderer, &lt9);
            SDL_Rect r9 = {x + size - thickness, y, thickness, size};
            SDL_RenderFillRect(renderer, &r9);
            break;
        }
    }
}


// Interactive drawing window
void interactive_draw_window(NeuralNetwork* nn) {
    const int CANVAS_SIZE = 28;
    const int SCALE = 20;  // Scale factor for display
    const int WINDOW_WIDTH = CANVAS_SIZE * SCALE + 300;
    const int WINDOW_HEIGHT = CANVAS_SIZE * SCALE + 100;
    
    // Initialize SDL
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        printf("SDL initialization failed: %s\n", SDL_GetError());
        return;
    }
    
    // Create window
    SDL_Window* window = SDL_CreateWindow(
        "Number Drawing - Press C to Clear, ESC to Exit",
        SDL_WINDOWPOS_UNDEFINED,
        SDL_WINDOWPOS_UNDEFINED,
        WINDOW_WIDTH,
        WINDOW_HEIGHT,
        SDL_WINDOW_SHOWN
    );
    
    if (!window) {
        printf("Failed to create window: %s\n", SDL_GetError());
        SDL_Quit();
        return;
    }
    
    SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
    if (!renderer) {
        printf("Failed to create renderer: %s\n", SDL_GetError());
        SDL_DestroyWindow(window);
        SDL_Quit();
        return;
    }
    
    // Canvas data (28x28 = 784 pixels)
    unsigned char canvas[CANVAS_SIZE * CANVAS_SIZE];
    memset(canvas, 0, sizeof(canvas));
    
    int mouse_down = 0;
    int quit = 0;
    int last_prediction = -1;
    int prediction_counter = 0;
    SDL_Event e;
    
    printf("\n=== Drawing Window Opened ===\n");
    printf("Instructions:\n");
    printf("  - Drag with left mouse button to draw numbers\n");
    printf("  - Press 'C' to clear canvas\n");
    printf("  - Press ESC or close window to exit\n");
    printf("  - Prediction results are displayed in real-time\n\n");
    
    while (!quit) {
        while (SDL_PollEvent(&e)) {
            if (e.type == SDL_QUIT) {
                quit = 1;
            } else if (e.type == SDL_KEYDOWN) {
                if (e.key.keysym.sym == SDLK_ESCAPE) {
                    quit = 1;
                } else if (e.key.keysym.sym == SDLK_c) {
                    // Clear canvas
                    memset(canvas, 0, sizeof(canvas));
                    printf("Canvas cleared\n");
                }
            } else if (e.type == SDL_MOUSEBUTTONDOWN) {
                if (e.button.button == SDL_BUTTON_LEFT) {
                    mouse_down = 1;
                }
            } else if (e.type == SDL_MOUSEBUTTONUP) {
                if (e.button.button == SDL_BUTTON_LEFT) {
                    mouse_down = 0;
                }
            } else if (e.type == SDL_MOUSEMOTION && mouse_down) {
                // Draw on canvas
                int mx = e.motion.x / SCALE;
                int my = e.motion.y / SCALE;
                
                if (mx >= 0 && mx < CANVAS_SIZE && my >= 0 && my < CANVAS_SIZE) {
                    // Draw a 2x2 brush
                    for (int dy = 0; dy < 2; dy++) {
                        for (int dx = 0; dx < 2; dx++) {
                            int px = mx + dx;
                            int py = my + dy;
                            if (px >= 0 && px < CANVAS_SIZE && py >= 0 && py < CANVAS_SIZE) {
                                int idx = py * CANVAS_SIZE + px;
                                if (canvas[idx] < 255) {
                                    canvas[idx] = 255;  // Set to white (inverted for MNIST)
                                }
                            }
                        }
                    }
                }
            }
        }
        
        // Clear screen
        SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
        SDL_RenderClear(renderer);
        
        // Draw canvas
        for (int y = 0; y < CANVAS_SIZE; y++) {
            for (int x = 0; x < CANVAS_SIZE; x++) {
                unsigned char pixel = canvas[y * CANVAS_SIZE + x];
                SDL_Rect rect = {
                    x * SCALE,
                    y * SCALE,
                    SCALE,
                    SCALE
                };
                SDL_SetRenderDrawColor(renderer, pixel, pixel, pixel, 255);
                SDL_RenderFillRect(renderer, &rect);
            }
        }
        
        // Draw grid lines
        SDL_SetRenderDrawColor(renderer, 50, 50, 50, 255);
        for (int i = 0; i <= CANVAS_SIZE; i++) {
            SDL_RenderDrawLine(renderer, i * SCALE, 0, i * SCALE, CANVAS_SIZE * SCALE);
            SDL_RenderDrawLine(renderer, 0, i * SCALE, CANVAS_SIZE * SCALE, i * SCALE);
        }
        
        // Real-time prediction (update every 10 frames to reduce CPU usage)
        prediction_counter++;
        if (prediction_counter >= 10) {
            prediction_counter = 0;
            double inputs[INPUT_NODES];
            for (int i = 0; i < INPUT_NODES; i++) {
                inputs[i] = canvas[i] / 255.0;
            }
            last_prediction = predict(nn, inputs);
        }
        
        // Draw prediction info area
        SDL_Rect info_rect = {
            CANVAS_SIZE * SCALE + 10,
            10,
            280,
            150
        };
        SDL_SetRenderDrawColor(renderer, 30, 30, 30, 255);
        SDL_RenderFillRect(renderer, &info_rect);
        SDL_SetRenderDrawColor(renderer, 100, 100, 100, 255);
        SDL_RenderDrawRect(renderer, &info_rect);
        
        // Draw prediction number using simple graphics (large digit)
        draw_digit_in_window(renderer, last_prediction, CANVAS_SIZE * SCALE + 120, 50, 100);
        
        // Update screen
        SDL_RenderPresent(renderer);
        
        // Small delay to prevent excessive CPU usage
        SDL_Delay(16);  // ~60 FPS
    }
    
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
    
    printf("Drawing window closed\n");
}


// Training function
void train_mode(NeuralNetwork* nn, CSVData* data) {
    printf("\n=== Training Mode ===\n");
    
    double inputs[INPUT_NODES];
    double targets[OUTPUT_NODES];
    
    int epochs = 3;
    printf("Starting training with %d epochs...\n", epochs);
    
    for (int epoch = 0; epoch < epochs; epoch++) {
        int correct = 0;
        for (int i = 0; i < data->rows; i++) {
            // Normalize input (0-255 -> 0.0-1.0)
            for (int j = 0; j < INPUT_NODES; j++) {
                inputs[j] = data->data[i][j+1] / 255.0;
            }
            // Target One-hot Encoding
            int label = data->data[i][0];
            for (int j = 0; j < OUTPUT_NODES; j++) targets[j] = 0.0;
            targets[label] = 1.0;

            train(nn, inputs, targets);

            // Calculate accuracy on the last epoch
            if (epoch == epochs - 1) {
                if (predict(nn, inputs) == label) correct++;
            }

            if (i % 1000 == 0) printf("Epoch %d: Progress %d/%d\r", epoch + 1, i, data->rows);
        }
        printf("\n");
    }
    
    // Save model after training
    save_model(nn, MODEL_FILENAME);
    printf("Training completed, model saved to %s\n", MODEL_FILENAME);
}

// Test function
void test_mode(NeuralNetwork* nn, CSVData* data, int total_samples) {
    printf("\n=== Test Mode ===\n");
    
    if (!load_model(nn, MODEL_FILENAME)) {
        printf("Error: Model file %s not found, please run training mode first\n", MODEL_FILENAME);
        return;
    }
    
    double inputs[INPUT_NODES];
    printf("Testing first %d samples...\n", total_samples);
    
    int correct = 0;
    for(int i = 0; i < total_samples && i < data->rows; i++) {
        for (int j = 0; j < INPUT_NODES; j++) {
            inputs[j] = data->data[i][j+1] / 255.0;
        }
        int prediction = predict(nn, inputs);
        int actual = data->data[i][0];
        
        const char* status = (actual == prediction) ? "Correct" : "Incorrect";
        if (actual == prediction) correct++;

        printf("Sample %02d: Actual = %d, Predicted = %d -> %s\n", i, actual, prediction, status);
    }

    printf("Accuracy: %.2f%%\n", (double)correct / total_samples * 100.0);
}

// Draw function
void draw_mode(NeuralNetwork* nn) {
    printf("\n=== Draw Mode ===\n");
    
    if (!load_model(nn, MODEL_FILENAME)) {
        printf("Error: Model file %s not found, please run training mode first\n", MODEL_FILENAME);
        return;
    }
    
    printf("Opening interactive drawing window...\n");
    interactive_draw_window(nn);
}

void print_usage(const char* program_name) {
    printf("Usage: %s [mode]\n", program_name);
    printf("\nMode options:\n");
    printf("  train, t     - Training mode: Train and save model (if no model file exists)\n");
    printf("  test, e      - Test mode: Test samples from train.csv (requires model file)\n");
    printf("  draw, d      - Draw mode: Open interactive drawing window (requires model file)\n");
    printf("\nExamples:\n");
    printf("  %s train    # Train model\n", program_name);
    printf("  %s test     # Test samples\n", program_name);
    printf("  %s draw     # Open drawing window\n", program_name);
}

int main(int argc, char* argv[]) {
    srand(time(NULL));
    
    // Parse command line arguments
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }
    
    char* mode = argv[1];
    
    // Initialize network structure
    NeuralNetwork nn;
    init_network(&nn);
    
    CSVData* data = NULL;
    
    // Load CSV data if needed (for train or test mode)
    if (strcmp(mode, "train") == 0 || strcmp(mode, "t") == 0 || 
        strcmp(mode, "test") == 0 || strcmp(mode, "e") == 0) {
        printf("Reading CSV data...\n");
        data = read_csv("train.csv");
        if (data == NULL) {
            printf("Error: Cannot read train.csv\n");
            free_network(&nn);
            return 1;
        }
        printf("Data read complete: %d rows, %d columns\n", data->rows, data->cols);
    }
    
    // Execute corresponding mode
    if (strcmp(mode, "train") == 0 || strcmp(mode, "t") == 0) {
        train_mode(&nn, data);
    } else if (strcmp(mode, "test") == 0 || strcmp(mode, "e") == 0) {
        int total_samples = 5;
        if (argc >= 3) {
            total_samples = atoi(argv[2]);
        }
        test_mode(&nn, data, total_samples);
    } else if (strcmp(mode, "draw") == 0 || strcmp(mode, "d") == 0) {
        draw_mode(&nn);
    } else {
        printf("Error: Unknown mode '%s'\n\n", mode);
        print_usage(argv[0]);
        free_network(&nn);
        if (data) free_csv(data);
        return 1;
    }
    
    // Cleanup
    free_network(&nn);
    if (data) free_csv(data);
    
    return 0;
}