plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}
val releaseKeystorePath = System.getenv("ANDROID_KEYSTORE_PATH")

android {
    namespace = "com.example.facepattern"
    compileSdk = 36
    defaultConfig {
        applicationId = "com.example.facepattern"
        minSdk = 24
        targetSdk = 36
        versionCode = 12
        versionName = "0.9.3"
        // Self-use target: OPPO Find X7 / modern ARM64 Android phones.
        // Keeping only this ABI avoids bundling unused x86 and 32-bit native runtimes.
        ndk {
            abiFilters += listOf("arm64-v8a")
        }
    }
    signingConfigs {
        create("release") {
            if (!releaseKeystorePath.isNullOrBlank()) {
                storeFile = file(releaseKeystorePath)
                storePassword = System.getenv("ANDROID_KEYSTORE_PASSWORD").orEmpty()
                keyAlias = System.getenv("ANDROID_KEY_ALIAS").orEmpty()
                keyPassword = System.getenv("ANDROID_KEY_PASSWORD").orEmpty()
            }
        }
    }
    buildTypes {
        getByName("release") {
            isMinifyEnabled = false
            if (!releaseKeystorePath.isNullOrBlank()) {
                signingConfig = signingConfigs.getByName("release")
            }
        }
    }
    compileOptions { sourceCompatibility = JavaVersion.VERSION_17; targetCompatibility = JavaVersion.VERSION_17 }
    kotlinOptions { jvmTarget = "17" }
}
dependencies {
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("androidx.camera:camera-camera2:1.4.2")
    implementation("androidx.camera:camera-lifecycle:1.4.2")
    implementation("androidx.camera:camera-view:1.4.2")
    implementation("androidx.exifinterface:exifinterface:1.3.7")
    implementation("com.google.mediapipe:tasks-vision:0.10.21")
    implementation("org.tensorflow:tensorflow-lite:2.17.0")
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.23.2")
}
