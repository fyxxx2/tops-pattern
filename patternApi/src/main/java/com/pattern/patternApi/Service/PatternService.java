package com.pattern.patternApi.Service;

import org.springframework.http.*;
import org.springframework.http.client.MultipartBodyBuilder;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.util.Map;

@Service
public class PatternService {

    private final RestTemplate restTemplate = new RestTemplate();
    private final String FASTAPI_URL = "http://localhost:8000/predict";

    public Map<String, Object> predict(MultipartFile file) {

        try {
            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.MULTIPART_FORM_DATA);

            // 이미지 파일을 FastAPI로 전송
            MultipartBodyBuilder builder = new MultipartBodyBuilder();
            builder.part("file", file.getBytes())
                    .header("Content-Disposition",
                            "form-data; name=file; filename=" + file.getOriginalFilename());

            HttpEntity<?> request = new HttpEntity<>(builder.build(), headers);

            ResponseEntity<Map> response = restTemplate.exchange(
                    FASTAPI_URL,
                    HttpMethod.POST,
                    request,
                    Map.class
            );

            return response.getBody();

        } catch (IOException e) {
            throw new RuntimeException("이미지 처리 중 오류 발생", e);
        }
    }
}