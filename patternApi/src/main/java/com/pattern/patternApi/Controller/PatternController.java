package com.pattern.patternApi.Controller;

import com.pattern.patternApi.Dto.PatternResponse;
import com.pattern.patternApi.Service.PatternService;
import lombok.RequiredArgsConstructor;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

@RestController
@RequiredArgsConstructor
@RequestMapping("/pattern")
public class PatternController {

    private final PatternService patternService;

    @PostMapping(value = "/predict", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public PatternResponse predict(@RequestParam("file") MultipartFile file) {
        return patternService.predictPattern(file);  // ✔ 객체로 호출
    }
}