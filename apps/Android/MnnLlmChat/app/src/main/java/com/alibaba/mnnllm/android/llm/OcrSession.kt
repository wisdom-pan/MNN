// Created by MNN on 2026/06/25.
// Copyright (c) 2026 Alibaba Group Holding Limited All rights reserved.

package com.alibaba.mnnllm.android.llm

import android.util.Log
import com.alibaba.mnnllm.android.chat.model.ChatDataItem
import com.alibaba.mnnllm.android.llm.ChatService.Companion.provide

/**
 * OCR session backed by the native PP-OCR detection + recognition engine.
 *
 * Unlike LLM/diffusion sessions, OCR is single-shot: [generate] takes an image
 * path (via params["imageInput"]) and returns the recognized text. There is no
 * streaming and no conversation history that the native layer cares about.
 */
class OcrSession(
    private val modelId: String,
    override var sessionId: String,
    private val configPath: String,
    private var savedHistory: List<ChatDataItem>? = null
) : ChatSession {
    override var supportOmni: Boolean = false
    override val debugInfo: String = ""
    private var nativePtr: Long = 0

    @Volatile
    private var releaseRequested = false

    @Volatile
    private var generating = false

    override fun load() {
        nativePtr = initNative(configPath, "")
        Log.d(TAG, "OcrSession load nativePtr=$nativePtr configPath=$configPath")
        if (releaseRequested) {
            release()
        }
    }

    override fun generate(
        prompt: String,
        params: Map<String, Any>,
        progressListener: GenerateProgressListener
    ): HashMap<String, Any> {
        synchronized(this) {
            if (nativePtr == 0L) {
                Log.e(TAG, "OCR nativePtr is 0, cannot generate")
                return hashMapOf(
                    "error" to true,
                    "message" to "Native OCR session not initialized"
                )
            }
            val imagePath = (params["imageInput"] as? String).orEmpty()
            if (imagePath.isEmpty()) {
                return hashMapOf(
                    "error" to true,
                    "message" to "OCR requires an image input"
                )
            }
            generating = true
            val nativeResult = runOcrNative(nativePtr, imagePath)
            val result: HashMap<String, Any> = nativeResult ?: hashMapOf(
                "error" to true,
                "message" to "Native OCR returned null"
            )
            // Stream the recognized text back so it renders as the assistant reply.
            (result["text"] as? String)?.let { text ->
                progressListener.onProgress(text)
            }
            progressListener.onProgress(null)
            generating = false
            if (releaseRequested) {
                releaseInner()
            }
            return result
        }
    }

    private fun releaseInner() {
        if (nativePtr != 0L) {
            releaseNative(nativePtr)
            nativePtr = 0
            provide().removeSession(sessionId)
            (this as Object).notifyAll()
        }
    }

    override fun reset(): String {
        this.sessionId = System.currentTimeMillis().toString()
        return this.sessionId
    }

    override fun release() {
        synchronized(this) {
            Log.d(TAG, "OcrSession release nativePtr: $nativePtr generating: $generating")
            if (!generating) {
                releaseInner()
            } else {
                releaseRequested = true
            }
        }
    }

    override fun setKeepHistory(keepHistory: Boolean) {}

    override fun setEnableAudioOutput(enable: Boolean) {}

    override fun getHistory(): List<ChatDataItem>? = savedHistory

    override fun setHistory(history: List<ChatDataItem>?) {
        savedHistory = history
    }

    override fun updateThinking(thinking: Boolean) {}

    private external fun initNative(configPath: String, extraConfig: String): Long

    private external fun runOcrNative(instanceId: Long, imagePath: String): HashMap<String, Any>?

    private external fun releaseNative(instanceId: Long)

    companion object {
        const val TAG: String = "OcrSession"

        init {
            System.loadLibrary("mnnllmapp")
        }
    }
}
