"""Foreground Service Builder for DroidBot-GPT framework.

This module provides utilities to create and manage foreground services
that help keep target apps in the foreground during automation.
"""

import os
import tempfile
import subprocess
from typing import Optional
from pathlib import Path

from ..core.logger import log


class ForegroundServiceBuilder:
    """Builder for creating foreground service APKs."""
    
    def __init__(self):
        """Initialize the foreground service builder."""
        self.temp_dir: Optional[str] = None
        self.apk_path: Optional[str] = None
    
    async def create_foreground_service_apk(self, target_package: str) -> Optional[str]:
        """Create a foreground service APK for the target app.
        
        Args:
            target_package: Package name of the target app.
            
        Returns:
            str: Path to the created APK, or None if failed.
        """
        try:
            # Create temporary directory
            self.temp_dir = tempfile.mkdtemp(prefix="droidbot_fg_service_")
            log.info(f"Created temporary directory: {self.temp_dir}")
            
            # Create APK structure
            await self._create_apk_structure(target_package)
            
            # Build the APK
            apk_path = await self._build_apk()
            
            if apk_path and os.path.exists(apk_path):
                self.apk_path = apk_path
                log.success(f"Foreground service APK created: {apk_path}")
                return apk_path
            else:
                log.error("Failed to build APK")
                return None
                
        except Exception as e:
            log.error(f"Failed to create foreground service APK: {e}")
            return None
    
    async def _create_apk_structure(self, target_package: str) -> None:
        """Create the APK directory structure and files.
        
        Args:
            target_package: Package name of the target app.
        """
        if not self.temp_dir:
            raise RuntimeError("Temporary directory not created")
        
        # Create main APK directory
        apk_dir = os.path.join(self.temp_dir, "foreground_service")
        os.makedirs(apk_dir, exist_ok=True)
        
        # Create AndroidManifest.xml
        await self._create_android_manifest(apk_dir, target_package)
        
        # Create Java source files
        await self._create_java_sources(apk_dir)
        
        # Create resource files
        await self._create_resources(apk_dir)
        
        # Create build files
        await self._create_build_files(apk_dir)
    
    async def _create_android_manifest(self, apk_dir: str, target_package: str) -> None:
        """Create AndroidManifest.xml file.
        
        Args:
            apk_dir: APK directory path.
            target_package: Target package name.
        """
        manifest_content = f'''<?xml version="1.0" encoding="utf-8"?>
<manifest xmlns:android="http://schemas.android.com/apk/res/android"
    package="com.droidbot.foregroundservice"
    android:versionCode="1"
    android:versionName="1.0">
    
    <uses-permission android:name="android.permission.FOREGROUND_SERVICE" />
    <uses-permission android:name="android.permission.POST_NOTIFICATIONS" />
    <uses-permission android:name="android.permission.WAKE_LOCK" />
    
    <application
        android:allowBackup="true"
        android:icon="@mipmap/ic_launcher"
        android:label="DroidBot Foreground Service"
        android:theme="@style/AppTheme">
        
        <service
            android:name=".ForegroundService"
            android:enabled="true"
            android:exported="false"
            android:foregroundServiceType="dataSync" />
            
        <activity
            android:name=".MainActivity"
            android:exported="true">
            <intent-filter>
                <action android:name="android.intent.action.MAIN" />
                <category android:name="android.intent.category.LAUNCHER" />
            </intent-filter>
        </activity>
            
    </application>
</manifest>'''
        
        manifest_path = os.path.join(apk_dir, "AndroidManifest.xml")
        with open(manifest_path, 'w') as f:
            f.write(manifest_content)
        
        log.debug("Created AndroidManifest.xml")
    
    async def _create_java_sources(self, apk_dir: str) -> None:
        """Create Java source files for the foreground service.
        
        Args:
            apk_dir: APK directory path.
        """
        # Create src directory
        src_dir = os.path.join(apk_dir, "src", "main", "java", "com", "droidbot", "foregroundservice")
        os.makedirs(src_dir, exist_ok=True)
        
        # Create MainActivity.java
        main_activity_content = '''package com.droidbot.foregroundservice;

import android.app.Activity;
import android.content.Intent;
import android.os.Bundle;

public class MainActivity extends Activity {
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        
        // Start the foreground service
        Intent serviceIntent = new Intent(this, ForegroundService.class);
        startForegroundService(serviceIntent);
        
        // Finish the activity
        finish();
    }
}'''
        
        main_activity_path = os.path.join(src_dir, "MainActivity.java")
        with open(main_activity_path, 'w') as f:
            f.write(main_activity_content)
        
        # Create ForegroundService.java
        service_content = '''package com.droidbot.foregroundservice;

import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.Service;
import android.content.Intent;
import android.os.Build;
import android.os.IBinder;
import androidx.core.app.NotificationCompat;

public class ForegroundService extends Service {
    private static final String CHANNEL_ID = "droidbot_automation";
    private static final int NOTIFICATION_ID = 1001;
    
    @Override
    public void onCreate() {
        super.onCreate();
        createNotificationChannel();
    }
    
    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        Notification notification = createNotification();
        startForeground(NOTIFICATION_ID, notification);
        return START_STICKY;
    }
    
    @Override
    public IBinder onBind(Intent intent) {
        return null;
    }
    
    private void createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            NotificationChannel channel = new NotificationChannel(
                CHANNEL_ID,
                "DroidBot Automation",
                NotificationManager.IMPORTANCE_LOW
            );
            channel.setDescription("Keeps app in foreground during automation");
            
            NotificationManager manager = getSystemService(NotificationManager.class);
            if (manager != null) {
                manager.createNotificationChannel(channel);
            }
        }
    }
    
    private Notification createNotification() {
        NotificationCompat.Builder builder = new NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle("DroidBot Automation")
            .setContentText("Keeping app in foreground for automation")
            .setSmallIcon(android.R.drawable.ic_dialog_info)
            .setOngoing(true)
            .setPriority(NotificationCompat.PRIORITY_LOW);
        
        return builder.build();
    }
}'''
        
        service_path = os.path.join(src_dir, "ForegroundService.java")
        with open(service_path, 'w') as f:
            f.write(service_content)
        
        log.debug("Created Java source files")
    
    async def _create_resources(self, apk_dir: str) -> None:
        """Create resource files for the APK.
        
        Args:
            apk_dir: APK directory path.
        """
        # Create res directory structure
        res_dir = os.path.join(apk_dir, "src", "main", "res")
        os.makedirs(res_dir, exist_ok=True)
        
        # Create values directory
        values_dir = os.path.join(res_dir, "values")
        os.makedirs(values_dir, exist_ok=True)
        
        # Create strings.xml
        strings_content = '''<?xml version="1.0" encoding="utf-8"?>
<resources>
    <string name="app_name">DroidBot Foreground Service</string>
</resources>'''
        
        strings_path = os.path.join(values_dir, "strings.xml")
        with open(strings_path, 'w') as f:
            f.write(strings_content)
        
        # Create styles.xml
        styles_content = '''<?xml version="1.0" encoding="utf-8"?>
<resources>
    <style name="AppTheme" parent="android:Theme.Material.Light">
        <item name="android:colorPrimary">#2196F3</item>
        <item name="android:colorPrimaryDark">#1976D2</item>
    </style>
</resources>'''
        
        styles_path = os.path.join(values_dir, "styles.xml")
        with open(styles_path, 'w') as f:
            f.write(styles_content)
        
        log.debug("Created resource files")
    
    async def _create_build_files(self, apk_dir: str) -> None:
        """Create build configuration files.
        
        Args:
            apk_dir: APK directory path.
        """
        # Create build.gradle
        build_gradle_content = '''plugins {
    id 'com.android.application'
}

android {
    compileSdk 33
    
    defaultConfig {
        applicationId "com.droidbot.foregroundservice"
        minSdk 21
        targetSdk 33
        versionCode 1
        versionName "1.0"
    }
    
    buildTypes {
        release {
            minifyEnabled false
            proguardFiles getDefaultProguardFile('proguard-android-optimize.txt'), 'proguard-rules.pro'
        }
    }
    
    compileOptions {
        sourceCompatibility JavaVersion.VERSION_1_8
        targetCompatibility JavaVersion.VERSION_1_8
    }
}

dependencies {
    implementation 'androidx.appcompat:appcompat:1.6.1'
    implementation 'androidx.core:core:1.10.1'
}'''
        
        build_gradle_path = os.path.join(apk_dir, "build.gradle")
        with open(build_gradle_path, 'w') as f:
            f.write(build_gradle_content)
        
        # Create settings.gradle
        settings_gradle_content = '''rootProject.name = "ForegroundService"'''
        
        settings_gradle_path = os.path.join(apk_dir, "settings.gradle")
        with open(settings_gradle_path, 'w') as f:
            f.write(settings_gradle_content)
        
        log.debug("Created build files")
    
    async def _build_apk(self) -> Optional[str]:
        """Build the APK using Gradle.
        
        Returns:
            str: Path to the built APK, or None if failed.
        """
        if not self.temp_dir:
            return None
        
        try:
            apk_dir = os.path.join(self.temp_dir, "foreground_service")
            
            # Check if we have Android SDK and Gradle available
            if not self._check_build_tools():
                log.warning("Android build tools not available, using simplified approach")
                return await self._create_simple_apk()
            
            # Build with Gradle
            result = subprocess.run(
                ["./gradlew", "assembleDebug"],
                cwd=apk_dir,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                # Find the built APK
                apk_path = os.path.join(apk_dir, "build", "outputs", "apk", "debug", "foreground_service-debug.apk")
                if os.path.exists(apk_path):
                    return apk_path
            
            log.error(f"Gradle build failed: {result.stderr}")
            return None
            
        except Exception as e:
            log.error(f"APK build failed: {e}")
            return None
    
    def _check_build_tools(self) -> bool:
        """Check if Android build tools are available.
        
        Returns:
            bool: True if build tools are available, False otherwise.
        """
        try:
            # Check for Android SDK
            android_home = os.environ.get('ANDROID_HOME')
            if not android_home:
                return False
            
            # Check for Gradle
            result = subprocess.run(["gradle", "--version"], capture_output=True, text=True)
            return result.returncode == 0
            
        except Exception:
            return False
    
    async def _create_simple_apk(self) -> Optional[str]:
        """Create a simple APK using alternative methods.
        
        Returns:
            str: Path to the created APK, or None if failed.
        """
        # For now, return None to indicate we should use shell-based approach
        # In a full implementation, you could use tools like apktool or aapt
        return None
    
    def cleanup(self) -> None:
        """Clean up temporary files."""
        try:
            if self.temp_dir and os.path.exists(self.temp_dir):
                import shutil
                shutil.rmtree(self.temp_dir)
                log.debug(f"Cleaned up temporary directory: {self.temp_dir}")
        except Exception as e:
            log.warning(f"Failed to cleanup temporary files: {e}")


# Global instance
_foreground_service_builder: Optional[ForegroundServiceBuilder] = None


def get_foreground_service_builder() -> ForegroundServiceBuilder:
    """Get the global foreground service builder instance.
    
    Returns:
        ForegroundServiceBuilder: The builder instance.
    """
    global _foreground_service_builder
    if _foreground_service_builder is None:
        _foreground_service_builder = ForegroundServiceBuilder()
    return _foreground_service_builder 