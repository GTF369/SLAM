package com.example.facepattern

import org.json.JSONArray
import org.json.JSONObject
import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import java.io.DataInputStream
import java.io.DataOutputStream
import java.nio.charset.StandardCharsets
import java.security.SecureRandom
import java.util.zip.ZipEntry
import java.util.zip.ZipInputStream
import java.util.zip.ZipOutputStream
import javax.crypto.Cipher
import javax.crypto.SecretKeyFactory
import javax.crypto.spec.GCMParameterSpec
import javax.crypto.spec.PBEKeySpec
import javax.crypto.spec.SecretKeySpec

/**
 * Portable, password-protected case backup codec.
 *
 * Case images and mask files are plaintext only inside the in-memory ZIP before the whole
 * archive is encrypted. Android Keystore keys are never exported. The receiving device
 * re-encrypts every imported case file with its own Keystore key.
 */
internal object PortableCaseBackup {
    private val MAGIC = "BROWPORT1".toByteArray(StandardCharsets.US_ASCII)
    private const val KDF_ITERATIONS = 210_000
    private const val SALT_BYTES = 16
    private const val IV_BYTES = 12
    private const val TAG_BITS = 128
    private const val MIN_PASSWORD_LENGTH = 12
    const val MAX_ARCHIVE_BYTES = 64 * 1024 * 1024
    const val MAX_ENCRYPTED_BUNDLE_BYTES = 65 * 1024 * 1024
    private const val MAX_ENTRIES = 500
    private const val MAX_MANIFEST_BYTES = 2 * 1024 * 1024

    internal data class Archive(
        val manifest: JSONObject,
        val files: Map<String, ByteArray>
    )

    fun referencedFiles(manifest: JSONObject): Set<String> {
        require(manifest.optInt("schema_version", 0) == 1) {
            "不支持的案例版本，无法生成可移植备份"
        }
        val result = linkedSetOf<String>()

        fun addRequired(name: String) {
            require(isSafeCaseFileName(name) && name.endsWith(".enc")) {
                "案例包含不支持或不安全的文件名"
            }
            result.add(name)
        }

        val photoName = manifest.optString("photo_file", "")
        addRequired(photoName)

        fun addMask(raw: Any?) {
            if (raw == null || raw == JSONObject.NULL) return
            val mask = raw as? JSONObject
                ?: throw IllegalArgumentException("案例中的识别轮廓索引格式无效")
            val name = mask.optString("mask_file", "")
            if (name.isNotBlank()) addRequired(name)
        }

        addMask(manifest.opt("current_segmentation"))
        val plans = manifest.optJSONArray("plans") ?: JSONArray()
        for (index in 0 until plans.length()) {
            val plan = plans.optJSONObject(index) ?: continue
            addMask(plan.opt("mask"))
        }
        require(result.isNotEmpty()) { "案例没有可备份的照片文件" }
        return result
    }

    fun pack(manifest: JSONObject, files: Map<String, ByteArray>): ByteArray {
        val requiredFiles = referencedFiles(manifest)
        require(files.keys == requiredFiles) {
            "备份文件与案例索引不一致"
        }
        val manifestBytes = manifest.toString().toByteArray(StandardCharsets.UTF_8)
        require(manifestBytes.size <= MAX_MANIFEST_BYTES) { "案例索引过大" }

        var uncompressedSize = manifestBytes.size.toLong()
        files.forEach { (name, bytes) ->
            require(isSafeCaseFileName(name) && name in requiredFiles) { "备份包含无效文件名" }
            require(bytes.isNotEmpty()) { "案例文件为空：$name" }
            uncompressedSize += bytes.size.toLong()
        }
        require(uncompressedSize <= MAX_ARCHIVE_BYTES) { "案例过大，请先移除不需要的方案或图片" }

        val buffer = ByteArrayOutputStream()
        ZipOutputStream(buffer).use { zip ->
            zip.putNextEntry(ZipEntry("manifest.json").apply { time = 0L })
            zip.write(manifestBytes)
            zip.closeEntry()
            files.toSortedMap().forEach { (name, bytes) ->
                zip.putNextEntry(ZipEntry("data/$name").apply { time = 0L })
                zip.write(bytes)
                zip.closeEntry()
            }
        }
        val archive = buffer.toByteArray()
        require(archive.size <= MAX_ARCHIVE_BYTES) { "压缩后的案例包过大" }
        return archive
    }

    fun unpack(zipBytes: ByteArray): Archive {
        require(zipBytes.isNotEmpty() && zipBytes.size <= MAX_ARCHIVE_BYTES) {
            "备份数据大小无效"
        }
        var manifestBytes: ByteArray? = null
        val files = linkedMapOf<String, ByteArray>()
        var totalInflated = 0L
        var entryCount = 0

        ZipInputStream(ByteArrayInputStream(zipBytes)).use { zip ->
            while (true) {
                val entry = zip.nextEntry ?: break
                entryCount++
                require(entryCount <= MAX_ENTRIES) { "备份包含过多文件" }
                require(!entry.isDirectory) { "备份格式包含不支持的目录项" }

                val name = entry.name
                val isManifest = name == "manifest.json"
                val dataName = if (name.startsWith("data/")) name.removePrefix("data/") else ""
                require(isManifest || (dataName.isNotBlank() && isSafeCaseFileName(dataName))) {
                    "备份包含不安全的 ZIP 路径"
                }
                require(if (isManifest) manifestBytes == null else dataName !in files) {
                    "备份包含重复文件"
                }

                val limit = if (isManifest) MAX_MANIFEST_BYTES else MAX_ARCHIVE_BYTES
                val bytes = ByteArrayOutputStream()
                val chunk = ByteArray(8192)
                while (true) {
                    val count = zip.read(chunk)
                    if (count < 0) break
                    totalInflated += count.toLong()
                    require(totalInflated <= MAX_ARCHIVE_BYTES) { "解压后的案例包超过大小限制" }
                    require(bytes.size().toLong() + count <= limit.toLong()) { "备份中的单个文件过大" }
                    bytes.write(chunk, 0, count)
                }
                if (isManifest) manifestBytes = bytes.toByteArray()
                else files[dataName] = bytes.toByteArray()
                zip.closeEntry()
            }
        }

        val serializedManifest = manifestBytes ?: throw IllegalArgumentException("备份缺少案例索引")
        val manifest = JSONObject(String(serializedManifest, StandardCharsets.UTF_8))
        val expectedFiles = referencedFiles(manifest)
        require(files.keys == expectedFiles) {
            "备份缺少案例文件，或包含案例索引之外的文件"
        }
        return Archive(manifest, files)
    }

    fun encrypt(archiveBytes: ByteArray, password: CharArray): ByteArray {
        requirePassword(password)
        require(archiveBytes.isNotEmpty() && archiveBytes.size <= MAX_ARCHIVE_BYTES) {
            "待加密的案例包大小无效"
        }
        val random = SecureRandom()
        val salt = ByteArray(SALT_BYTES).also(random::nextBytes)
        val iv = ByteArray(IV_BYTES).also(random::nextBytes)
        val keyBytes = deriveKey(password, salt, KDF_ITERATIONS)
        try {
            val cipher = Cipher.getInstance("AES/GCM/NoPadding")
            cipher.init(Cipher.ENCRYPT_MODE, SecretKeySpec(keyBytes, "AES"), GCMParameterSpec(TAG_BITS, iv))
            val encrypted = cipher.doFinal(archiveBytes)
            val result = ByteArrayOutputStream()
            DataOutputStream(result).use { output ->
                output.write(MAGIC)
                output.writeInt(KDF_ITERATIONS)
                output.writeByte(salt.size)
                output.writeByte(iv.size)
                output.write(salt)
                output.write(iv)
                output.write(encrypted)
            }
            val payload = result.toByteArray()
            require(payload.size <= MAX_ENCRYPTED_BUNDLE_BYTES) { "加密后的案例包过大" }
            return payload
        } finally {
            keyBytes.fill(0)
        }
    }

    fun decrypt(payload: ByteArray, password: CharArray): ByteArray {
        requirePassword(password)
        require(payload.size in (MAGIC.size + 4 + 2 + SALT_BYTES + IV_BYTES + 16)..MAX_ENCRYPTED_BUNDLE_BYTES) {
            "文件不是受支持的眉形案例备份，或文件大小无效"
        }
        val input = DataInputStream(ByteArrayInputStream(payload))
        val receivedMagic = ByteArray(MAGIC.size)
        input.readFully(receivedMagic)
        require(receivedMagic.contentEquals(MAGIC)) { "备份格式不受支持" }

        val iterations = input.readInt()
        require(iterations in 100_000..500_000) { "备份的密钥派生参数不安全或不受支持" }
        val saltLength = input.readUnsignedByte()
        val ivLength = input.readUnsignedByte()
        require(saltLength == SALT_BYTES && ivLength == IV_BYTES) { "备份文件头损坏" }
        val salt = ByteArray(saltLength)
        val iv = ByteArray(ivLength)
        input.readFully(salt)
        input.readFully(iv)
        val encrypted = input.readBytes()
        require(encrypted.size >= 16) { "加密内容不完整" }

        val keyBytes = deriveKey(password, salt, iterations)
        try {
            val cipher = Cipher.getInstance("AES/GCM/NoPadding")
            cipher.init(Cipher.DECRYPT_MODE, SecretKeySpec(keyBytes, "AES"), GCMParameterSpec(TAG_BITS, iv))
            val plaintext = cipher.doFinal(encrypted)
            require(plaintext.size <= MAX_ARCHIVE_BYTES) { "解密后的案例包超过大小限制" }
            return plaintext
        } finally {
            keyBytes.fill(0)
        }
    }

    private fun deriveKey(password: CharArray, salt: ByteArray, iterations: Int): ByteArray {
        val spec = PBEKeySpec(password, salt, iterations, 256)
        return try {
            SecretKeyFactory.getInstance("PBKDF2WithHmacSHA256").generateSecret(spec).encoded
        } finally {
            spec.clearPassword()
        }
    }

    private fun requirePassword(password: CharArray) {
        require(password.size >= MIN_PASSWORD_LENGTH) {
            "备份口令至少需要 12 个字符"
        }
    }

    private fun isSafeCaseFileName(name: String): Boolean =
        name.isNotBlank() &&
            name.length <= 180 &&
            name != "." &&
            name != ".." &&
            !name.contains("..") &&
            !name.contains('/') &&
            !name.contains('\\') &&
            name.none { it.code < 0x20 } &&
            java.io.File(name).name == name
}
