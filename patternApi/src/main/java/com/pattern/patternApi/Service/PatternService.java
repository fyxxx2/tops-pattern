package com.pattern.patternApi.Service;

import com.pattern.patternApi.Dto.PatternResponse;
import lombok.RequiredArgsConstructor;
import org.springframework.core.io.ByteArrayResource;
import org.springframework.stereotype.Service;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.client.RestTemplate;
import org.springframework.http.*;
import java.util.HashMap;
import java.util.Map;

@Service
@RequiredArgsConstructor
public class PatternService {

    private final RestTemplate restTemplate = new RestTemplate();

    public PatternResponse predictPattern(MultipartFile file) {

        try {
            String fastApiUrl = "http://127.0.0.1:8000/predict";

            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.MULTIPART_FORM_DATA);

            // 파일을 보낼 form-data 구성
            MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
            body.add("file", new ByteArrayResource(file.getBytes()) {
                @Override
                public String getFilename() {
                    return file.getOriginalFilename();
                }
            });

            HttpEntity<MultiValueMap<String, Object>> request = new HttpEntity<>(body, headers);

            ResponseEntity<Map> response = restTemplate.postForEntity(fastApiUrl, request, Map.class);

            // FastAPI가 반환한 값
            Map responseBody = response.getBody();

            return new PatternResponse(
                    (String) responseBody.get("prediction"),
                    (Double) responseBody.get("confidence")
            );

        } catch (Exception e) {
            return new PatternResponse("error", 0.0);
        }
    }
}