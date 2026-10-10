package com.example.rag;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.*;
import java.net.URI;
import java.net.http.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.time.Instant;
import java.util.*;
import java.util.stream.Collectors;

public class PromptToChunksToOllama {

    private static final ObjectMapper M = new ObjectMapper();

    public static void main(String[] args) throws Exception {
        Map<String, String> cfg = parseArgs(args);
        String promptFile = cfg.getOrDefault("prompt_file", "input/prompt.txt");
        String chunksCsv = cfg.getOrDefault("chunks_csv", "chunks.csv");
        String ollamaUrl = cfg.getOrDefault("ollama_url", "http://localhost:11434");
        String embedModel = cfg.getOrDefault("embed_model", "nomic-embed-text");
        String llmModel = cfg.getOrDefault("llm_model", "llama3.2");
        int topK = Integer.parseInt(cfg.getOrDefault("top_k", "5"));
        int maxTokens = Integer.parseInt(cfg.getOrDefault("max_tokens", "512"));
        Path outputDir = Paths.get(cfg.getOrDefault("output_dir", "output"));

        // 1) Read prompt
        String prompt = Files.readString(Path.of(promptFile), StandardCharsets.UTF_8).trim();
        if (prompt.isEmpty()) {
            System.err.println("Prompt file is empty: " + promptFile);
            System.exit(2);
        }
        System.out.println("Loaded prompt (len=" + prompt.length() + ")");

        HttpClient http = HttpClient.newHttpClient();

        // 2) Embedding
        double[] promptEmbedding = embedTextOllama(http, ollamaUrl, embedModel, prompt);
        System.out.println("Prompt embedding length: " + promptEmbedding.length);

        // 3) Load chunks + score
        List<ChunkRow> results = readChunksCsvAndScore(chunksCsv, promptEmbedding);
        if (results.isEmpty()) {
            System.err.println("No valid chunk embeddings found in CSV or embedding dimension mismatch.");
            System.exit(3);
        }

        List<ChunkRow> topChunks = results.stream()
                .sorted(Comparator.comparingDouble((ChunkRow r) -> r.score).reversed())
                .limit(topK)
                .collect(Collectors.toList());

        System.out.println("Top " + topK + " chunks:");
        for (int i = 0; i < topChunks.size(); i++) {
            ChunkRow c = topChunks.get(i);
            System.out.printf("%d) doc=%s idx=%d score=%.6f snippet=\"%s\"%n",
                    i+1, c.docId, c.chunkIndex, c.score,
                    (c.chunkText.length() > 120 ? c.chunkText.substring(0,120) + "..." : c.chunkText));
        }

        // 4) Build LLM prompt
        String llmPrompt = buildJsonEnforcedPrompt(prompt, topChunks, maxTokens);

        // 5) Call Ollama CLI
        String llmOutput = callOllamaCompletionViaCLI(llmModel, llmPrompt);

        // -------- CLEAN OUTPUT -----------
        String cleanedOutput = sanitizeCliOutput(llmOutput);
        Files.writeString(outputDir.resolve("raw_llm_output_cleaned.txt"), cleanedOutput, StandardCharsets.UTF_8);

        // 6) Extract JSON from cleaned output
        Files.createDirectories(outputDir);
        String extractedJson = extractFirstJsonObject(cleanedOutput);

        if (extractedJson == null) {
            System.err.println("Warning: No JSON object found in cleaned LLM output. Saved files for debugging.");
            Files.writeString(outputDir.resolve("raw_llm_output.txt"), llmOutput, StandardCharsets.UTF_8);
            Files.writeString(outputDir.resolve("cleaned_output_no_json.txt"), cleanedOutput, StandardCharsets.UTF_8);
            System.exit(0);
        } else {
            try {
                JsonNode resNode = parseAndValidateResult(extractedJson);
                String baseName = "result_" + Instant.now().toEpochMilli();

                writeStructuredOutput(resNode, outputDir, baseName);

                // Pretty extracted JSON
                Files.writeString(
                        outputDir.resolve(baseName + "_extracted_pretty.json"),
                        M.writerWithDefaultPrettyPrinter().writeValueAsString(resNode),
                        StandardCharsets.UTF_8
                );

                System.out.println("Structured result written to " + outputDir.toAbsolutePath());
            } catch (Exception e) {
                Files.writeString(outputDir.resolve("raw_llm_output.txt"), llmOutput, StandardCharsets.UTF_8);
                Files.writeString(outputDir.resolve("extracted_json.txt"), extractedJson, StandardCharsets.UTF_8);
                System.err.println("Failed to parse JSON: " + e.getMessage());
                System.exit(1);
            }
        }
    }

    // --------------------------------------------------------------------
    //  SANITIZER — Removes ANSI escape codes and illegal control characters
    // --------------------------------------------------------------------
    static String sanitizeCliOutput(String s) {
        if (s == null || s.isEmpty()) return s;

        String ansiRegex =
                "(?s)\\u001B\\[[0-9;?]*[ -/]*[@-~]" +    // CSI sequences
                "|\\u001B\\][^\\u0007]*(\\u0007)?" +    // OSC sequences
                "|\\u001B.[\\s\\S]?";                   // Other ESC

        s = s.replaceAll(ansiRegex, "");

        // Remove bad control chars except newline, tab, CR
        s = s.replaceAll("[\\x00-\\x08\\x0B\\x0C\\x0E-\\x1F\\x7F]", "");

        s = s.replaceAll("\\r\\n", "\n");
        s = s.replaceAll("\\n{3,}", "\n\n");

        return s.trim();
    }

    // --------------------------------------------------------------------
    // Build strict JSON-output prompt
    // --------------------------------------------------------------------
    static String buildJsonEnforcedPrompt(String userPrompt, List<ChunkRow> topChunks, int maxTokens) {
        StringBuilder sb = new StringBuilder();
        sb.append("SYSTEM: You are a SailPoint transform/rule generator. ");
        sb.append("You MUST return exactly one valid JSON object and nothing else. ");
        sb.append("Temperature must behave as 0.0 — produce deterministic output.\n\n");

        sb.append("OUTPUT SCHEMA:\n");
        sb.append("{\n");
        sb.append("  \"code\": \"<string>\",\n");
        sb.append("  \"language\": \"<groovy|javascript|java|xml>\",\n");
        sb.append("  \"tests\": [ { \"input\": {...}, \"expected\": {...} } ],\n");
        sb.append("  \"explanation\": \"<short explanation>\",\n");
        sb.append("  \"provenance\": [ { \"doc_id\":\"...\", \"score\":0.9, \"snippet\":\"...\" } ]\n");
        sb.append("}\n\n");

        sb.append("Use only the following retrieved chunks:\n");
        for (ChunkRow c : topChunks) {
            sb.append("---- chunk ").append(c.docId).append("#").append(c.chunkIndex)
                    .append(" score=").append(c.score).append(" ----\n");
            sb.append(c.chunkText).append("\n\n");
        }

        sb.append("USER PROMPT:\n").append(userPrompt).append("\n\n");
        sb.append("CONSTRAINTS:\n");
        sb.append("- Produce concise output (~").append(maxTokens).append(" tokens)\n");
        sb.append("- No markdown, no commentary outside JSON\n");
        sb.append("- If unable, return {\"error\":\"reason\"}\n");

        return sb.toString();
    }

    // --------------------------------------------------------------------
    // Embeddings API call
    // --------------------------------------------------------------------
    static double[] embedTextOllama(HttpClient http, String ollamaUrl, String model, String text) throws Exception {
        URI uri = URI.create(ollamaUrl.replaceAll("/$", "") + "/v1/embeddings");

        Map<String, Object> payload = new HashMap<>();
        payload.put("model", model);
        payload.put("input", text);

        HttpRequest req = HttpRequest.newBuilder(uri)
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(M.writeValueAsString(payload)))
                .build();

        HttpResponse<String> resp = http.send(req, HttpResponse.BodyHandlers.ofString());
        if (resp.statusCode() != 200)
            throw new RuntimeException("Embedding failed: " + resp.body());

        JsonNode j = M.readTree(resp.body());
        JsonNode emb = j.get("data").get(0).get("embedding");

        double[] vec = new double[emb.size()];
        for (int i = 0; i < emb.size(); i++) vec[i] = emb.get(i).asDouble();
        return vec;
    }

    // --------------------------------------------------------------------
    // CLI LLM call (stdin-based, no flags)
    // --------------------------------------------------------------------
    static String callOllamaCompletionViaCLI(String model, String prompt) throws Exception {
        List<String> cmd = List.of("ollama", "run", model);
        ProcessBuilder pb = new ProcessBuilder(cmd);
        pb.redirectErrorStream(true);
        Process proc = pb.start();

        // write prompt
        try (BufferedWriter writer = new BufferedWriter(
                new OutputStreamWriter(proc.getOutputStream(), StandardCharsets.UTF_8))) {
            writer.write(prompt);
        }

        // read output
        StringBuilder out = new StringBuilder();
        try (BufferedReader reader = new BufferedReader(
                new InputStreamReader(proc.getInputStream(), StandardCharsets.UTF_8))) {
            String line;
            while ((line = reader.readLine()) != null) out.append(line).append("\n");
        }

        proc.waitFor(240, java.util.concurrent.TimeUnit.SECONDS);

        if (proc.exitValue() != 0)
            throw new RuntimeException("Ollama CLI error: " + out.toString());

        return out.toString();
    }

    // --------------------------------------------------------------------
    // CSV loading + scoring
    // --------------------------------------------------------------------
    static List<ChunkRow> readChunksCsvAndScore(String csv, double[] promptEmb) throws Exception {
        List<ChunkRow> list = new ArrayList<>();
        try (BufferedReader br = Files.newBufferedReader(Path.of(csv), StandardCharsets.UTF_8)) {
            String header = br.readLine();
            if (header == null) throw new RuntimeException("Empty CSV");

            String line;
            while ((line = br.readLine()) != null) {
                List<String> cols = parseCsvLine(line);
                if (cols.size() < 7) continue;

                String docId = unquote(cols.get(1));
                int idx = Integer.parseInt(cols.get(2));
                String text = unquote(cols.get(3));
                String embJson = unquote(cols.get(5));

                double[] emb = jsonArrayToDoubleArray(embJson);
                if (emb.length != promptEmb.length) continue;

                double score = cosineSimilarity(promptEmb, emb);
                list.add(new ChunkRow(docId, idx, text, emb, score));
            }
        }
        return list;
    }

    // --------------------------------------------------------------------
    // JSON extraction
    // --------------------------------------------------------------------
    static String extractFirstJsonObject(String text) {
        int start = text.indexOf('{');
        while (start >= 0) {
            int depth = 0;
            boolean inStr = false;

            for (int i = start; i < text.length(); i++) {
                char c = text.charAt(i);
                if (c == '"' && (i == 0 || text.charAt(i - 1) != '\\')) inStr = !inStr;

                if (!inStr) {
                    if (c == '{') depth++;
                    else if (c == '}') depth--;

                    if (depth == 0)
                        return text.substring(start, i + 1);
                }
            }
            start = text.indexOf('{', start + 1);
        }
        return null;
    }

    static JsonNode parseAndValidateResult(String json) throws Exception {
        JsonNode n = M.readTree(json);
        if (!n.isObject()) throw new IllegalArgumentException("JSON is not an object");
        if (!n.has("code") || !n.has("language"))
            throw new IllegalArgumentException("Missing required fields");

        return n;
    }

    // --------------------------------------------------------------------
    // Write structured output
    // --------------------------------------------------------------------
    static void writeStructuredOutput(JsonNode node, Path out, String base) throws Exception {
        Files.createDirectories(out);

        Files.writeString(
                out.resolve(base + "_result.json"),
                M.writerWithDefaultPrettyPrinter().writeValueAsString(node),
                StandardCharsets.UTF_8
        );

        // code file
        String lang = node.get("language").asText();
        String ext = switch (lang.toLowerCase()) {
            case "groovy" -> ".groovy";
            case "javascript" -> ".js";
            case "xml" -> ".xml";
            case "java" -> ".java";
            default -> ".txt";
        };

        Files.writeString(
                out.resolve(base + "_code" + ext),
                node.get("code").asText(),
                StandardCharsets.UTF_8
        );

        // tests
        if (node.has("tests"))
            Files.writeString(
                    out.resolve(base + "_tests.json"),
                    M.writerWithDefaultPrettyPrinter().writeValueAsString(node.get("tests")),
                    StandardCharsets.UTF_8
            );

        // provenance
        if (node.has("provenance"))
            Files.writeString(
                    out.resolve(base + "_provenance.json"),
                    M.writerWithDefaultPrettyPrinter().writeValueAsString(node.get("provenance")),
                    StandardCharsets.UTF_8
            );
    }

    // --------------------------------------------------------------------
    // CSV helpers
    // --------------------------------------------------------------------
    static List<String> parseCsvLine(String line) {
        List<String> cols = new ArrayList<>();
        boolean inQuote = false;
        StringBuilder sb = new StringBuilder();

        for (char c : line.toCharArray()) {
            if (c == '"') {
                inQuote = !inQuote;
            } else if (c == ',' && !inQuote) {
                cols.add(sb.toString());
                sb.setLength(0);
            } else {
                sb.append(c);
            }
        }
        cols.add(sb.toString());
        return cols;
    }

    static String unquote(String s) {
        if (s.startsWith("\"") && s.endsWith("\""))
            return s.substring(1, s.length() - 1).replace("\"\"", "\"");
        return s;
    }

    static double[] jsonArrayToDoubleArray(String json) throws Exception {
        JsonNode arr = M.readTree(json);
        double[] a = new double[arr.size()];
        for (int i = 0; i < arr.size(); i++) a[i] = arr.get(i).asDouble();
        return a;
    }

    static double cosineSimilarity(double[] a, double[] b) {
        double dot = 0, na = 0, nb = 0;
        for (int i = 0; i < a.length; i++) {
            dot += a[i] * b[i];
            na += a[i] * a[i];
            nb += b[i] * b[i];
        }
        return dot / (Math.sqrt(na) * Math.sqrt(nb));
    }

    static Map<String,String> parseArgs(String[] args) {
        Map<String,String> m = new HashMap<>();
        for (int i = 0; i < args.length; i++) {
            if (args[i].startsWith("--") && i+1 < args.length) {
                m.put(args[i].substring(2), args[++i]);
            }
        }
        return m;
    }

    // --------------------------------------------------------------------
    // Data class
    // --------------------------------------------------------------------
    static class ChunkRow {
        String docId;
        int chunkIndex;
        String chunkText;
        double[] embedding;
        double score;

        ChunkRow(String docId, int chunkIndex, String chunkText, double[] embedding, double score) {
            this.docId = docId;
            this.chunkIndex = chunkIndex;
            this.chunkText = chunkText;
            this.embedding = embedding;
            this.score = score;
        }
    }
}
