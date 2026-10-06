# Capturing a small object

Photogrammetry finds the same points of the object in many photos and
works out where each photo was taken from. Everything below serves that:
the same detail must be visible, sharp and lit the same way in several
photos.

## The setup

- **Light: bright and soft.** Overcast daylight, or two lamps through
  diffusers (or bounced off a white wall). Avoid hard shadows and the
  on-camera flash: shadows and highlights that move with you look like
  different surfaces.
- **A patterned mat under the object.** Newspaper, a printed pattern, a
  textured cloth. It gives COLMAP points to lock onto around the object,
  which helps most with plain objects. It gets reconstructed too; the
  object is still the part in focus.
- **A plain or blurred background beyond the mat** is fine. Don't let
  people or things move in the background between photos.
- **Fill the frame** with the object (and a bit of mat): two-thirds of the
  photo or more.

## Taking the photos

- **Walk all the way around**, a photo every 10 to 15° (24 to 36 photos per
  circle). Keep a similar distance.
- **Two heights at least:** one ring about level with the object (10-20°
  above the table) and one from higher up (30-45°), so the top is seen. Add
  close-ups of fine detail last.
- **Every part in at least three photos**, with neighbouring photos
  overlapping by about two-thirds.
- **Sharp photos:** good light, a fast enough shutter (1/125 s or faster
  hand-held), and enough depth of field (f/8 to f/11 on a camera; phones are
  usually fine). The photo checks flag the blurry ones.
- **Don't zoom** during a set, and keep the same camera and lens. With a
  zoom lens, tape the ring. If a set mixes cameras or zoom settings, each
  is calibrated on its own automatically, but one per set is more accurate.
- **Keep EXIF data** (copy the original files, don't export or resize):
  the focal length in it helps a lot.
- **iPhone:** HEIC photos are fine: a JPEG copy of each is made on import
  (the original is kept).

## Video instead of photos

Video works for a quick capture: walk slowly around the object in good
light, 30 seconds to 2 minutes. EZ2DIGITIZE keeps the sharpest frame of
each stretch (100 frames by default). Photos still give finer detail.

## Turntables

With the camera on a tripod and the object turning, the background stays
still and wins: COLMAP places the cameras from the background, so they all
end up in one spot (the coverage check says so after camera placement).
Use masks: in the **Masks** tab, **Make masks** separates the object from
the background in every photo (about a second per photo; `ez2d masks P`
on the command line). On the skull turntable set this is the difference
between a smeared shell and a clean model. Look through the grid: a mask
that cut off part of the object is worse than none, so uncheck it. A
plain cloth behind and under the object, lit evenly, helps both the masks
and the reconstruction. Masks made elsewhere (one PNG per photo, white for
the object) can be added with **Add masks…** or `ez2d import P photos/
--masks masks/`.

## Both sides of an object

To capture the underside, take a full set, turn the object over, and take a
second set. Import both and make masks for every photo (so only the
object, not the table, is matched); both sets then join into one model. A guided flip
workflow is planned.

## Difficult objects

- **Shiny:** reflections move with the camera and break matching. Use a
  removable matte spray (scanning spray, or dry shampoo/foot powder for a
  quick test), or cross-polarized light.
- **Transparent or very dark:** same, matte spray.
- **No texture (plain white plastic):** the patterned mat helps the camera
  placement; for the surface itself, a light pattern (masking tape bits,
  washable marker, projected dots) gives the matcher something to find.
- **Thin parts** (wires, hair, leaves): they will be thick or missing at
  any quality; High helps a little.

## After a run

The log notes what the reconstruction saw: how many photos were placed,
gaps around the object, all photos from one height, photos placed far from
the rest. They are hints for the next capture.
