import os
from datetime import datetime
import json
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import cv2
from PIL import ImageEnhance

class Config:
    DATA_DIR = './flower/flower_images/'
    MODEL_NAME = 'model1.keras'
    IMG_SIZE = (128, 128)
    BATCH_SIZE = 32
    EPOCHS = 100
    PATIENCE = 20
    NUM_CLASSES = 5
    LEARNING_RATE = 0.001

    CLASS_INFO = {
        'daisy': {'emoji': '🌼', 'description': 'Simple white petals with yellow center'},
        'lavender': {'emoji': '💜', 'description': 'Purple spikes with a calming fragrance'},
        'lotus': {'emoji': '🪷', 'description': 'Sacred water lily with large round leaves'},
        'sunflower': {'emoji': '🌻', 'description': 'Large yellow petals with dark center'},
        'tulip': {'emoji': '🌷', 'description': 'Cup-shaped flower with smooth petals'}
    }


def create_data_generators():
    print("Creating data generator with advanced augmentation...")

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        rotation_range=30,
        width_shift_range=0.3,
        height_shift_range=0.3,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.8, 1.2],
        channel_shift_range=0.2,
        fill_mode='nearest'
    )

    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )

    train_generator = train_datagen.flow_from_directory(
        Config.DATA_DIR,
        target_size=Config.IMG_SIZE,
        batch_size=Config.BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True,
        seed=42
    )

    val_generator = val_datagen.flow_from_directory(
        Config.DATA_DIR,
        target_size=Config.IMG_SIZE,
        batch_size=Config.BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=True,
        seed=42
    )
    print(f"✅ Training samples: {train_generator.samples}")
    print(f"✅ Validation samples: {val_generator.samples}")
    print(f"✅ Class indices: {train_generator.class_indices}")
    return train_generator, val_generator

def build_enhanced_model():
    """Build an enhanced CNN model with residual connections"""
    print("🏗️ Building enhanced CNN model...")

    inputs = layers.Input(shape=(128, 128, 3))

    # First block
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    # Second block with residual connection
    shortcut = x
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)

    # Residual connection
    shortcut = layers.Conv2D(64, (1, 1), padding='same')(shortcut)
    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    # Third block
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    # Fourth block
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)

    # Dense layers
    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)

    # Output layer
    outputs = layers.Dense(Config.NUM_CLASSES, activation='softmax')(x)

    model = models.Model(inputs, outputs)

    # Compile with custom optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=Config.LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy', 'top_k_categorical_accuracy']
    )

    print("✅ Model built successfully!")
    print(f"📊 Total parameters: {model.count_params():,}")

    return model

def create_callbacks():
    """Create enhanced callbacks for training"""
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=Config.PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        ModelCheckpoint(
            Config.MODEL_NAME,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]

    return callbacks

def evaluate_model_comprehensive(model, validation_generator):
    """Comprehensive model evaluation with detailed metrics"""
    print("🔍 Performing comprehensive model evaluation...")

    # Get class names
    class_indices = validation_generator.class_indices
    class_names = [k.title() for k in sorted(class_indices.keys(), key=class_indices.get)]

    # Predict on validation data
    validation_generator.reset()
    y_pred = []
    y_true = []
    y_pred_proba = []

    print("📊 Generating predictions...")
    for i in range(len(validation_generator)):
        x, y = validation_generator[i]
        pred_proba = model.predict(x, verbose=0)
        y_pred.extend(np.argmax(pred_proba, axis=1))
        y_true.extend(np.argmax(y, axis=1))
        y_pred_proba.extend(pred_proba)

    y_pred_proba = np.array(y_pred_proba)

    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Create enhanced confusion matrix plot
    plt.figure(figsize=(12, 10))

    # Calculate percentages
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    # Create annotations
    annot = np.empty_like(cm).astype(str)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f'{cm[i, j]}\n({cm_percent[i, j]:.1f}%)'

    sns.heatmap(cm,
                annot=annot,
                fmt='',
                cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names,
                cbar_kws={'label': 'Count'})

    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('Enhanced Confusion Matrix\n(Count and Percentage)', fontsize=14)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Generate classification report
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)

    print("\n📈 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    # Calculate additional metrics
    accuracy = report['accuracy']
    macro_avg_f1 = report['macro avg']['f1-score']
    weighted_avg_f1 = report['weighted avg']['f1-score']

    # Per-class confidence analysis
    confidence_analysis = {}
    for i, class_name in enumerate(class_names):
        class_predictions = y_pred_proba[np.array(y_true) == i]
        if len(class_predictions) > 0:
            confidence_analysis[class_name] = {
                'mean_confidence': np.mean(np.max(class_predictions, axis=1)),
                'std_confidence': np.std(np.max(class_predictions, axis=1)),
                'correct_predictions': np.sum(np.array(y_pred)[np.array(y_true) == i] == i)
            }

    # Save detailed metrics
    # Convert NumPy values to native Python types to ensure JSON serialization works
    metrics = {
        'accuracy': float(accuracy),
        'macro_avg_f1': float(macro_avg_f1),
        'weighted_avg_f1': float(weighted_avg_f1),
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
        'confidence_analysis': confidence_analysis,
        'evaluation_date': datetime.now().isoformat()
    }

    # Convert any remaining NumPy values in nested dictionaries
    def convert_to_serializable(obj):
        if isinstance(obj, np.number):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(i) for i in obj]
        else:
            return obj

    metrics = convert_to_serializable(metrics)

    with open('model_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"✅ Overall Accuracy: {accuracy:.4f}")
    print(f"✅ Macro Average F1: {macro_avg_f1:.4f}")
    print(f"✅ Weighted Average F1: {weighted_avg_f1:.4f}")

    return metrics

def create_advanced_plots(history, metrics):
    """Create advanced training plots"""
    print("📊 Creating advanced visualization plots...")

    # Training history plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Accuracy plot
    axes[0, 0].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
    axes[0, 0].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    axes[0, 0].set_title('Model Accuracy Over Time', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Loss plot
    axes[0, 1].plot(history.history['loss'], label='Training Loss', linewidth=2)
    axes[0, 1].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0, 1].set_title('Model Loss Over Time', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Learning rate plot (if available)
    if 'lr' in history.history:
        axes[1, 0].plot(history.history['lr'], linewidth=2, color='orange')
        axes[1, 0].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)

    # Top-K accuracy plot
    if 'top_k_categorical_accuracy' in history.history:
        axes[1, 1].plot(history.history['top_k_categorical_accuracy'],
                       label='Training Top-K Accuracy', linewidth=2)
        axes[1, 1].plot(history.history['val_top_k_categorical_accuracy'],
                       label='Validation Top-K Accuracy', linewidth=2)
        axes[1, 1].set_title('Top-K Categorical Accuracy', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Top-K Accuracy')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Per-class performance plot
    if 'confidence_analysis' in metrics:
        classes = list(metrics['confidence_analysis'].keys())
        confidences = [metrics['confidence_analysis'][c]['mean_confidence'] for c in classes]

        plt.figure(figsize=(10, 6))
        bars = plt.bar(classes, confidences, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8'])

        # Add value labels on bars
        for bar, conf in zip(bars, confidences):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{conf:.3f}', ha='center', va='bottom', fontweight='bold')

        plt.title('Mean Prediction Confidence by Class', fontsize=14, fontweight='bold')
        plt.xlabel('Flower Class')
        plt.ylabel('Mean Confidence')
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('class_confidence.png', dpi=300, bbox_inches='tight')
        plt.close()

def analyze_dataset():
    """Analyze the dataset and provide insights"""
    print("🔍 Analyzing dataset...")

    if not os.path.exists(Config.DATA_DIR):
        print(f"❌ Dataset directory not found: {Config.DATA_DIR}")
        return

    class_counts = {}
    total_images = 0

    for class_name in os.listdir(Config.DATA_DIR):
        class_path = os.path.join(Config.DATA_DIR, class_name)
        if os.path.isdir(class_path):
            count = len([f for f in os.listdir(class_path)
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            class_counts[class_name] = count
            total_images += count

    print(f"📊 Dataset Analysis:")
    print(f"   Total images: {total_images}")
    print(f"   Number of classes: {len(class_counts)}")
    print(f"   Class distribution:")

    for class_name, count in sorted(class_counts.items()):
        percentage = (count / total_images) * 100
        print(f"     {class_name}: {count} images ({percentage:.1f}%)")

    # Check for class imbalance
    min_count = min(class_counts.values())
    max_count = max(class_counts.values())
    imbalance_ratio = max_count / min_count

    if imbalance_ratio > 2:
        print(f"⚠️  Class imbalance detected (ratio: {imbalance_ratio:.2f})")
        print("   Consider using class weights or additional data augmentation")
    else:
        print(f"✅ Dataset is well balanced (ratio: {imbalance_ratio:.2f})")

    return class_counts

# Main training function
def train_enhanced_model():
    """Main training function with enhanced features"""
    print("🚀 Starting enhanced flower classifier training...")
    print("="*60)

    # Analyze dataset
    class_counts = analyze_dataset()
    if not class_counts:
        return

    # Create data generators
    train_generator, validation_generator = create_data_generators()

    # Build model
    model = build_enhanced_model()

    # Display model summary
    print("\n📋 Model Summary:")
    model.summary()

    # Create callbacks
    callbacks = create_callbacks()

    # Calculate steps per epoch
    steps_per_epoch = train_generator.samples // Config.BATCH_SIZE
    validation_steps = validation_generator.samples // Config.BATCH_SIZE

    print(f"\n🎯 Training Configuration:")
    print(f"   Epochs: {Config.EPOCHS}")
    print(f"   Batch size: {Config.BATCH_SIZE}")
    print(f"   Steps per epoch: {steps_per_epoch}")
    print(f"   Validation steps: {validation_steps}")
    print(f"   Learning rate: {Config.LEARNING_RATE}")

    # Train the model
    print("\n🏃‍♂️ Starting training...")
    start_time = datetime.now()

    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=Config.EPOCHS,
        validation_data=validation_generator,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1
    )

    end_time = datetime.now()
    training_time = end_time - start_time

    print(f"\n✅ Training completed in {training_time}")

    # Evaluate the model
    print("\n🔍 Evaluating model...")
    metrics = evaluate_model_comprehensive(model, validation_generator)

    # Create advanced plots
    create_advanced_plots(history, metrics)

    # Save training history
    # Convert history.history from a Keras History object to a serializable dict
    serializable_history = {}
    for key, value in history.history.items():
        serializable_history[key] = [float(v) for v in value]

    history_dict = {
        'history': serializable_history,
        'training_time': str(training_time),
        'final_accuracy': float(max(history.history['val_accuracy'])),
        'final_loss': float(min(history.history['val_loss'])),
        'total_epochs': len(history.history['accuracy'])
    }

    with open('training_history.json', 'w') as f:
        json.dump(history_dict, f, indent=2)

    print("\n🎉 Training completed successfully!")
    print("="*60)
    print("📄 Generated files:")
    print("   - flower_classifier.keras (trained model)")
    print("   - training_history.png (training plots)")
    print("   - confusion_matrix.png (confusion matrix)")
    print("   - class_confidence.png (per-class confidence)")
    print("   - model_metrics.json (detailed metrics)")
    print("   - training_history.json (training history)")
    print("\n🚀 You can now run the Streamlit app: streamlit run app.py")

if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Train the model
    train_enhanced_model()