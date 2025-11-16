import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np

# Configuration
BASE_PATH = 'C:/Users/thara/Downloads/dataset-20251108T035419Z-1-001/dataset'
IMG_SIZE = (224, 224)
BATCH_SIZE = 16  # Smaller batch for better generalization
EPOCHS = 100
INITIAL_LR = 0.0001

DATASET_CONFIG = {
    'path': os.path.join(BASE_PATH, 'Sugarcane'),
    'model_name': 'Sugarcane_resnet50.keras',
    'classes': ['Healthy', 'Mosaic', 'RedRot', 'Rust', 'Yellow']
}

def create_data_generators(train_dir, val_dir):
    """Create data generators with strong augmentation"""
    
    # Strong augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.25,
        height_shift_range=0.25,
        shear_range=0.2,
        zoom_range=0.25,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest'
    )
    
    # Only rescaling for validation
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )
    
    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    return train_generator, val_generator

def build_model(num_classes):
    """Build ResNet50 model with custom classification head"""
    
    # Load pre-trained ResNet50
    base_model = ResNet50(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze base model
    base_model.trainable = False
    
    # Build model with dropout and regularization
    inputs = keras.Input(shape=(*IMG_SIZE, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation='relu', 
                     kernel_regularizer=keras.regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs)
    
    return model, base_model

def plot_training_history(history, filename='Sugarcane_training_history.png'):
    """Plot and save training history"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Accuracy plot
    epochs_range = range(1, len(history.history['accuracy']) + 1)
    ax1.plot(epochs_range, history.history['accuracy'], 'b-', label='Training', linewidth=2)
    ax1.plot(epochs_range, history.history['val_accuracy'], 'r-', label='Validation', linewidth=2)
    ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1])
    
    # Loss plot
    ax2.plot(epochs_range, history.history['loss'], 'b-', label='Training', linewidth=2)
    ax2.plot(epochs_range, history.history['val_loss'], 'r-', label='Validation', linewidth=2)
    ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved plot: {filename}")

def evaluate_per_class(model, val_gen):
    """Evaluate per-class accuracy"""
    
    print("\n" + "="*70)
    print("PER-CLASS PERFORMANCE")
    print("="*70)
    
    # Get predictions
    y_true = val_gen.classes
    y_pred = model.predict(val_gen, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Get class names
    class_names = list(val_gen.class_indices.keys())
    
    print(f"\n{'Class':<30s} {'Correct':<10s} {'Total':<10s} {'Accuracy':<10s}")
    print("-" * 70)
    
    for i, class_name in enumerate(class_names):
        class_mask = y_true == i
        if class_mask.sum() > 0:
            class_correct = (y_pred_classes[class_mask] == i).sum()
            class_total = class_mask.sum()
            class_acc = class_correct / class_total
            
            # Color coding
            if class_acc >= 0.8:
                status = "✅"
            elif class_acc >= 0.6:
                status = "⚠️ "
            else:
                status = "❌"
            
            print(f"{status} {class_name:<28s} {class_correct:<10d} {class_total:<10d} {class_acc:>6.1%}")

def train_model():
    """Main training function"""
    
    print("\n" + "="*70)
    print("RICE DISEASE CLASSIFICATION - RESNET50")
    print("="*70)
    
    # Paths
    train_dir = os.path.join(DATASET_CONFIG['path'], 'train')
    val_dir = os.path.join(DATASET_CONFIG['path'], 'validation')
    
    # Verify paths
    if not os.path.exists(train_dir):
        print(f"❌ ERROR: Training directory not found: {train_dir}")
        return None, None
    if not os.path.exists(val_dir):
        print(f"❌ ERROR: Validation directory not found: {val_dir}")
        return None, None
    
    # Create generators
    print("\n📁 Loading datasets...")
    train_gen, val_gen = create_data_generators(train_dir, val_dir)
    
    num_classes = len(DATASET_CONFIG['classes'])
    print(f"\n✓ Data loaded successfully!")
    print(f"  Training samples:   {train_gen.samples}")
    print(f"  Validation samples: {val_gen.samples}")
    print(f"  Number of classes:  {num_classes}")
    print(f"  Classes: {DATASET_CONFIG['classes']}")
    
    # Build model
    print("\n🔨 Building ResNet50 model...")
    model, base_model = build_model(num_classes)
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=INITIAL_LR),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"\n✓ Model compiled!")
    print(f"  Total parameters: {model.count_params():,}")
    trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            DATASET_CONFIG['model_name'],
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_accuracy',
            patience=20,
            restore_best_weights=True,
            verbose=1,
            mode='max'
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # PHASE 1: Train with frozen base
    print("\n" + "="*70)
    print("PHASE 1: Transfer Learning (Frozen Base Model)")
    print("="*70)
    
    history1 = model.fit(
        train_gen,
        epochs=30,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1
    )
    
    plot_training_history(history1, 'Sugarcane_phase1_history.png')
    
    # Check if Phase 1 was successful
    best_val_acc_phase1 = max(history1.history['val_accuracy'])
    print(f"\n📊 Phase 1 Best Validation Accuracy: {best_val_acc_phase1:.4f} ({best_val_acc_phase1*100:.2f}%)")
    
    if best_val_acc_phase1 < 0.3:
        print("\n❌ Phase 1 failed to learn. Possible issues:")
        print("   1. Images might be corrupted or unreadable")
        print("   2. Labels might be incorrect")
        print("   3. Images might be too similar across classes")
        print("\n💡 Suggestion: Manually inspect images in each folder")
        return model, history1
    
    # PHASE 2: Fine-tuning (only if Phase 1 succeeded)
    if best_val_acc_phase1 >= 0.5:
        print("\n" + "="*70)
        print("PHASE 2: Fine-Tuning (Unfreezing Last Layers)")
        print("="*70)
        
        # Unfreeze last 30 layers
        base_model.trainable = True
        for layer in base_model.layers[:-30]:
            layer.trainable = False
        
        num_trainable = sum([1 for layer in base_model.layers if layer.trainable])
        print(f"\n✓ Unfroze {num_trainable} layers for fine-tuning")
        
        # Recompile with lower learning rate
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=INITIAL_LR/10),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Reset callbacks
        callbacks_phase2 = [
            ModelCheckpoint(
                DATASET_CONFIG['model_name'],
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            EarlyStopping(
                monitor='val_accuracy',
                patience=15,
                restore_best_weights=True,
                verbose=1,
                mode='max'
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=4,
                min_lr=1e-8,
                verbose=1
            )
        ]
        
        # Continue training
        history2 = model.fit(
            train_gen,
            epochs=EPOCHS,
            initial_epoch=len(history1.history['accuracy']),
            validation_data=val_gen,
            callbacks=callbacks_phase2,
            verbose=1
        )
        
        # Combine histories
        for key in history1.history.keys():
            history1.history[key] += history2.history[key]
        
        plot_training_history(history1, 'final_history.png')
    else:
        print("\n⚠️  Skipping Phase 2 - Phase 1 accuracy too low for fine-tuning")
    
    # Final Evaluation
    print("\n" + "="*70)
    print("FINAL EVALUATION")
    print("="*70)
    
    val_loss, val_accuracy = model.evaluate(val_gen, verbose=0)
    
    print(f"\n{'='*70}")
    print(f"  Final Validation Loss:     {val_loss:.4f}")
    print(f"  Final Validation Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
    print(f"{'='*70}")
    
    # Per-class evaluation
    evaluate_per_class(model, val_gen)
    
    # Save model
    model.save(DATASET_CONFIG['model_name'])
    print(f"\n✓ Model saved: {DATASET_CONFIG['model_name']}")
    
    # Final recommendations
    print("\n" + "="*70)
    print("DIAGNOSIS & RECOMMENDATIONS")
    print("="*70)
    
    if val_accuracy < 0.5:
        print("\n❌ POOR PERFORMANCE (<50%)")
        print("\nLikely issues:")
        print("  1. Disease symptoms too similar - hard to distinguish")
        print("  2. Image quality problems (blur, lighting, angles)")
        print("  3. Mislabeled training data")
        print("\nNext steps:")
        print("  • Manually review 20-30 random images from each class")
        print("  • Check if YOU can tell the diseases apart")
        print("  • Look for obviously mislabeled images")
        print("  • Consider merging similar disease classes")
        
    elif val_accuracy < 0.7:
        print("\n⚠️  MODERATE PERFORMANCE (50-70%)")
        print("\nSuggestions:")
        print("  • Collect more diverse training data")
        print("  • Try ensemble of multiple models")
        print("  • Use test-time augmentation")
        print("  • Focus on improving worst-performing classes")
        
    else:
        print("\n✅ GOOD PERFORMANCE (>70%)")
        print("\nNext steps:")
        print("  • Test on completely new data")
        print("  • Create prediction script for deployment")
        print("  • Monitor real-world performance")
    
    return model, history1

if __name__ == "__main__":
    # Check GPU
    print("\n" + "="*70)
    print("SYSTEM CHECK")
    print("="*70)
    
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"\n✓ GPU Available: {gpus[0].name}")
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
            print("✓ GPU memory growth enabled")
        except:
            pass
    else:
        print("\nℹ️  No GPU detected - using CPU")
        print("   Training will be slower (~5-10x)")
    
    print(f"\nTensorFlow version: {tf.__version__}")
    print(f"Keras version: {keras.__version__}")
    
    # Train
    model, history = train_model()
    
    if model is not None:
        print("\n" + "="*70)
        print("✅ TRAINING COMPLETE!")
        print("="*70)
    else:
        print("\n" + "="*70)
        print("❌ TRAINING INCOMPLETE")
        print("="*70)