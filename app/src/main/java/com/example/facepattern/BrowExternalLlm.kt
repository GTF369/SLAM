package com.example.facepattern

import org.json.JSONArray
import org.json.JSONObject
import java.net.HttpURLConnection
import java.net.URL

/**
 * Optional, user-triggered OpenAI-compatible Chat Completions adapter.
 * It sends numeric geometry and candidate parameters only, never the portrait or case ID.
 * The returned model score is an uncalibrated design opinion, not a customer preference probability.
 */
internal object BrowExternalLlm {
    fun analyze(
        baseUrl: String,
        model: String,
        apiKey: String,
        payload: JSONObject
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
            You are an eyebrow-design review assistant. You will receive only anonymized numerical
            facial-proportion features, candidate geometry parameters, and a stated style intent.
            You will NOT receive a face photograph or customer identity. Do not claim to know the
            customer's hidden tastes, beauty, ethnicity, health, or identity. Do not claim to calculate
            the probability that a real customer will like a candidate. Return a comparative engineering
            opinion only. Output concise Simplified Chinese with candidate style, a 0-100 heuristic
            design-compatibility score, two grounded observations tied to the supplied numerical input,
            and a clear uncertainty note. The score is not a probability and is not calibrated.
            Avoid inventing precise anatomy or making medical claims.
        """.trimIndent()
        val requestBody = JSONObject().apply {
            put("model", model.trim())
            put("temperature", 0.2)
            put("stream", false)
            put("messages", JSONArray()
                .put(JSONObject().put("role", "system").put("content", systemPrompt))
                .put(JSONObject().put("role", "user").put("content", payload.toString())))
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
