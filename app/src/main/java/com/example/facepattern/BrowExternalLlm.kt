package com.example.facepattern

import org.json.JSONArray
import org.json.JSONObject
import java.net.HttpURLConnection
import java.net.URL

/**
 * Optional, user-triggered OpenAI-compatible Chat Completions adapter.
 * A portrait is sent only when the user has explicitly accepted the external-processing consent.
 * Candidate parameters are sent without case ID, name, contact details, or other direct identifiers.
 * Returned scores are AI design opinions, never calibrated customer-preference probabilities.
 */
internal object BrowExternalLlm {
    fun analyze(
        baseUrl: String,
        model: String,
        apiKey: String,
        payload: JSONObject,
        imageDataUrl: String? = null
    ): String {
        val root = baseUrl.trim().trimEnd('/')
        val parsedEndpoint = try { URL(root) } catch (error: Exception) {
            throw IllegalArgumentException("Invalid endpoint URL.", error)
        }
        val isLocalHttp = parsedEndpoint.protocol == "http" &&
            parsedEndpoint.host.lowercase() in setOf("10.0.2.2", "127.0.0.1", "localhost")
        require(parsedEndpoint.protocol == "https" || isLocalHttp) {
            "Remote model endpoints must use HTTPS. Plain HTTP is allowed only for localhost development."
        }
        require(parsedEndpoint.userInfo == null) { "Do not embed credentials in the endpoint URL." }
        require(model.isNotBlank()) { "Please enter the model name." }
        val endpoint = if (root.endsWith("/chat/completions")) root else "$root/chat/completions"
        val systemPrompt = """
            You are an eyebrow-design review assistant. You may receive a face photo, anonymized facial
            proportion features, three candidate geometry parameter sets, and a stated design intent.
            Compare only the visible proportions and the supplied candidate parameters. Do not infer
            ethnicity, identity, health, personality, hidden tastes, or what the customer will choose.
            This is an AI-assisted design compatibility review, not customer preference ground truth.
            Return concise Simplified Chinese: for each candidate, provide a 0-100 relative design
            compatibility score on the same scale, two observations grounded in the visible image or
            supplied numbers, and uncertainty. Do not call any score a probability or imply it is
            calibrated. If the image is absent, explicitly note that the review used numerical features
            only. Do not invent precise anatomical measurements or make medical claims.
        """.trimIndent()
        val userContent: Any = if (!imageDataUrl.isNullOrBlank()) {
            JSONArray()
                .put(JSONObject().put("type", "text").put("text", payload.toString()))
                .put(JSONObject().put("type", "image_url").put(
                    "image_url", JSONObject().put("url", imageDataUrl).put("detail", "high")
                ))
        } else {
            payload.toString()
        }
        val requestBody = JSONObject().apply {
            put("model", model.trim())
            put("temperature", 0.2)
            put("stream", false)
            put("messages", JSONArray()
                .put(JSONObject().put("role", "system").put("content", systemPrompt))
                .put(JSONObject().put("role", "user").put("content", userContent)))
        }

        val connection = URL(endpoint).openConnection() as HttpURLConnection
        try {
            connection.requestMethod = "POST"
            connection.connectTimeout = 15_000
            connection.readTimeout = 45_000
            connection.doOutput = true
            connection.setRequestProperty("Content-Type", "application/json; charset=utf-8")
            connection.setRequestProperty("Accept", "application/json")
            if (apiKey.isNotBlank()) {
                connection.setRequestProperty("Authorization", "Bearer ${apiKey.trim()}")
            }
            connection.outputStream.use { it.write(requestBody.toString().toByteArray(Charsets.UTF_8)) }
            val status = connection.responseCode
            val stream = if (status in 200..299) connection.inputStream else connection.errorStream
            val raw = stream?.bufferedReader(Charsets.UTF_8)?.use { it.readText() }.orEmpty()
            if (status !in 200..299) {
                val brief = try { JSONObject(raw).optJSONObject("error")?.optString("message") } catch (_: Exception) { null }
                throw IllegalStateException("Provider HTTP $status" + (brief?.takeIf { it.isNotBlank() }?.let { ": $it" } ?: ""))
            }
            val response = JSONObject(raw)
            val content = response.optJSONArray("choices")
                ?.optJSONObject(0)
                ?.optJSONObject("message")
                ?.optString("content")
                ?.trim()
            if (content.isNullOrBlank()) throw IllegalStateException("The provider returned no text content.")
            return content
        } finally {
            connection.disconnect()
        }
    }
}
