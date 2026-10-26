# 50 questions for the range-pass lake

Type these in the **Ask** box on the L3M page (`/opt/mpl/frontend/next/index.html`). That page is the web front of the appliance. Leave Research and Teach off.

Each line was run through the same text door the chat uses, against `/opt/mpldatalake/stores/eworld`, on 25 Oct 2026. The note after the question is the top fact that came back. A class or a scene is on many frames, so the top fact is one of them, not the whole set. Ask the frame number to read that frame.

The door binds a question only when the words are a surface it already has. `shown in`, `depicted in`, `story setting`, `situated at`, `on-screen text`, `also known as`, `part of`, `point in time`, `running time`, and `attribute of` are those surfaces. A looser sentence such as "what does the truck depict" does not bind.

`person` and `train` are also ordinary words in other mounted lakes. `frame N`, `range pass`, `humvee`, `armor`, and `ground-with-person` are this lake.

## The film

1. What is the title of range pass?
   Opens `range pass`.
2. What is the running time of range pass?
   Opens `168500 ms`.
3. What is the length of time of range pass?
   Opens the same duration.
4. What is the source of description of range pass?
   Opens `yolov8+yolov8-world`.
5. What is range pass an aspect of?
   Opens `pass 2`.

## Which film a frame belongs to, and when

6. What is frame 0 part of?
   Opens `range pass`. This frame has no box.
7. What is frame 5 part of?
   Opens `range pass`.
8. What is frame 26 part of?
   Opens `range pass`.
9. What is frame 100 part of?
   Opens `range pass`.
10. What is frame 337 part of?
    Opens `range pass`. This is the last sampled frame.
11. What is the point in time of frame 0?
    Opens `0` milliseconds.
12. What is the point in time of frame 5?
    Opens `2500`.
13. What is the point in time of frame 26?
    Opens `13000`.
14. What is the point in time of frame 100?
    Opens `50000`.

## Scene of a frame, and where a scene sits

15. What is the story setting of frame 5?
    Opens `air`.
16. What is the story setting of frame 26?
    Opens `armor`.
17. What is the story setting of frame 100?
    Opens `armor`.
18. What is the story setting of frame 134?
    Opens `ground-with-person`.
19. What is the story setting of frame 158?
    Opens `ground`.
20. What is the place of action of frame 196?
    Opens `water`.
21. What is armor situated at?
    Opens `frame 100`.
22. What is air situated at?
    Opens `frame 153`.
23. What is water situated at?
    Opens `frame 196`.
24. What is ground situated at?
    Opens `frame 158`.
25. What is ground-with-person situated at?
    Opens `frame 134`.
26. What is person situated at?
    Opens `frame 115`. This is the scene named person, not a person detection.

## Where an object was seen

27. What is the truck shown in?
    Opens `frame 100`.
28. What is the truck portrayed by?
    Opens the same truck fact.
29. What is the airplane shown in?
    Opens `frame 146`.
30. What is the airplane rendered by?
    Opens the same airplane fact.
31. What is the humvee shown in?
    Opens `frame 101`.
32. What is the tank shown in?
    Opens `frame 147`.
33. What is the boat shown in?
    Opens `frame 112`.
34. What is the helicopter shown in?
    Opens `frame 131`.
35. What is the person shown in?
    Opens `frame 113`. This is a person detection.
36. What is the train shown in?
    Opens `frame 162`.
37. What is the artillery shown in?
    Opens `frame 78`.
38. What is the dining table shown in?
    Opens `frame 10`.
39. What is the stop sign shown in?
    Opens `frame 13`.
40. What is the parking meter shown in?
    Opens `frame 13`.

## What a frame depicts

41. What is frame 5 depicted in?
    Opens `airplane`.
42. What is the depiction of frame 5?
    Opens the same airplane fact.
43. What is frame 147 depicted in?
    Opens `humvee`. The tank is also on this frame. The top fact is the humvee.
44. What is the story setting of frame 146?
    Opens `armor`, on the same frame the airplane question names.

## On-screen text, repeated views, and the empty frame

45. What is the on-screen text of frame 5?
    Opens `new-track`. The index queued this frame for OCR.
46. What is the on-screen text of frame 26?
    Opens `tick`. The index queued it again on the same track.
47. What is frame 11 also known as?
    Opens `frame 10`. A later view of a track already queued.
48. What is frame 26 synonymous with?
    Opens `frame 20`.
49. What is frame 84 also known as?
    Opens `frame 81`. Frame 84 has no on-screen-text fact. It depicts a train, a traffic light, and a boat, and its story setting is water.
50. What is frame 0 an attribute of?
    Opens `no-box`. Frame 0 has no story setting and no on-screen text.

## Letters read from the OCR pool

The OCR pool is `eworld_ocr`, mounted beside the detection lake. Florence read the 135 frames `ocr.json` queued. 88 of them returned stencil text. A frame's title is that plate. The plate itself is shown in the frame.

On the detection lake, "on-screen text" of a frame is still the queue reason (`new-track` or `tick`). Ask the title of the frame, or ask the plate, to read the letters.

51. What is the title of ocr pass?
    Opens `ocr pass`.
52. What is the source of description of ocr pass?
    Opens `florence-2`.
53. What is ocr pass an aspect of?
    Opens `pass 2`.
54. What is the title of frame 26?
    Opens `B127`. The same frame also holds `NORTH` and `RIDGE`.
55. What is B127 shown in?
    Opens `frame 149`. B127 is the plate on many frames. This is the top one.
56. What is the on-screen text of B127?
    Opens `frame 149`.
57. What is the title of frame 12?
    Opens `RANGE`.
58. What is RANGE shown in?
    Opens `frame 12`.
59. What is the title of frame 20?
    Opens `NORTH`. The other token on that frame is `RODGE`.
60. What is RIDGE shown in?
    Opens `frame 26`.
61. What is the title of frame 46?
    Opens `C330`.
62. What is C330 shown in?
    Opens `frame 46`.
63. What is the title of frame 86?
    Opens `A22`.
64. What is A22 shown in?
    Opens `frame 86`.
65. What is the title of frame 98?
    Opens `A14`.
66. What is the title of frame 179?
    Opens `27`.
67. What is the title of frame 196?
    Opens `P19`.
68. What is P19 shown in?
    Opens `frame 196`.
69. What is the title of frame 226?
    Opens `AMMO`.
70. What is the title of frame 241?
    Opens `T441`.
71. What is T441 shown in?
    Opens `frame 241`.
72. What is the title of frame 293?
    Opens `AF218`.
73. What is GATE shown in?
    Opens `frame 310`. The title of that frame is `NORTH`.
74. What is the title of frame 316?
    Opens `FUEL`.
75. What is the title of frame 327?
    Opens `MK82`.
