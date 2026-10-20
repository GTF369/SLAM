plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}
android {
    namespace = "com.example.facepattern"
    compileSdk = 35
    defaultConfig {
        applicationId = "com.example.facepattern"
        minSdk = 24
        targetSdk = 35
        versionCode = 4
        versionName = "0.4.0"
    }
    buildTypes { release { isMinifyEnabled = false } }
    compileOptions { sourceCompatibility = JavaVersion.VERSION_17; targetCompatibility = JavaVersion.VERSION_17 }
    kotlinOptions { jvmTarget = "17" }
}
dependencies {
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("androidx.camera:camera-camera2:1.4.2")
    implementation("androidx.camera:camera-lifecycle:1.4.2")
    implementation("androidx.camera:camera-view:1.4.2")
    implementation("com.google.mediapipe:tasks-vision:0.10.21")
}
