//
// Created by MNN on 2026/06/25.
// Copyright (c) 2026 Alibaba Group Holding Limited All rights reserved.
//

#include <jni.h>

#include <sstream>
#include <string>

#include "mls_log.h"
#include "ocr_session.h"

using namespace mls;

extern "C" JNIEXPORT jlong JNICALL
Java_com_alibaba_mnnllm_android_llm_OcrSession_initNative(JNIEnv* env, jobject thiz,
                                                          jstring resource_path,
                                                          jstring config_json) {
    const char* resource_path_cstr = env->GetStringUTFChars(resource_path, nullptr);
    const char* config_cstr = env->GetStringUTFChars(config_json, nullptr);
    MNN_DEBUG("OcrSession_initNative resource: %s", resource_path_cstr);

    int backend_type = MNN_FORWARD_CPU;
    std::string config_str = config_cstr ? config_cstr : "";
    if (config_str.find("opencl") != std::string::npos) {
        backend_type = MNN_FORWARD_OPENCL;
    }

    auto* session = new OcrSession(resource_path_cstr, backend_type);
    session->Load();

    env->ReleaseStringUTFChars(resource_path, resource_path_cstr);
    env->ReleaseStringUTFChars(config_json, config_cstr);
    return reinterpret_cast<jlong>(session);
}

extern "C" JNIEXPORT void JNICALL
Java_com_alibaba_mnnllm_android_llm_OcrSession_releaseNative(JNIEnv* env, jobject thiz,
                                                             jlong instance_id) {
    auto* session = reinterpret_cast<OcrSession*>(instance_id);
    delete session;
}

extern "C" JNIEXPORT jobject JNICALL
Java_com_alibaba_mnnllm_android_llm_OcrSession_runOcrNative(JNIEnv* env, jobject thiz,
                                                            jlong instance_id,
                                                            jstring image_path) {
    jclass hashMapClass = env->FindClass("java/util/HashMap");
    jmethodID hashMapInit = env->GetMethodID(hashMapClass, "<init>", "()V");
    jmethodID putMethod =
        env->GetMethodID(hashMapClass, "put", "(Ljava/lang/Object;Ljava/lang/Object;)Ljava/lang/Object;");
    jobject hashMap = env->NewObject(hashMapClass, hashMapInit);
    jclass booleanClass = env->FindClass("java/lang/Boolean");
    jmethodID booleanInit = env->GetMethodID(booleanClass, "<init>", "(Z)V");

    auto putError = [&](const char* message) {
        env->CallObjectMethod(hashMap, putMethod, env->NewStringUTF("success"),
                              env->NewObject(booleanClass, booleanInit, JNI_FALSE));
        env->CallObjectMethod(hashMap, putMethod, env->NewStringUTF("error"),
                              env->NewObject(booleanClass, booleanInit, JNI_TRUE));
        env->CallObjectMethod(hashMap, putMethod, env->NewStringUTF("message"),
                              env->NewStringUTF(message));
    };

    auto* session = reinterpret_cast<OcrSession*>(instance_id);
    if (session == nullptr) {
        putError("Native OCR session not initialized");
        return hashMap;
    }

    const char* image_path_cstr = env->GetStringUTFChars(image_path, nullptr);
    std::string image_path_str = image_path_cstr ? image_path_cstr : "";
    env->ReleaseStringUTFChars(image_path, image_path_cstr);

    try {
        auto results = session->Run(image_path_str);
        // Concatenate recognized lines into a single text block.
        std::string text;
        for (size_t i = 0; i < results.size(); ++i) {
            text += results[i].text;
            if (i + 1 < results.size()) {
                text += "\n";
            }
        }
        env->CallObjectMethod(hashMap, putMethod, env->NewStringUTF("success"),
                              env->NewObject(booleanClass, booleanInit, JNI_TRUE));
        env->CallObjectMethod(hashMap, putMethod, env->NewStringUTF("text"),
                              env->NewStringUTF(text.c_str()));
        jclass integerClass = env->FindClass("java/lang/Integer");
        jmethodID integerInit = env->GetMethodID(integerClass, "<init>", "(I)V");
        env->CallObjectMethod(hashMap, putMethod, env->NewStringUTF("count"),
                              env->NewObject(integerClass, integerInit, (jint)results.size()));
    } catch (const std::exception& e) {
        MNN_ERROR("OcrSession runOcrNative exception: %s", e.what());
        putError(e.what());
    } catch (...) {
        putError("Unknown native OCR error");
    }
    return hashMap;
}
